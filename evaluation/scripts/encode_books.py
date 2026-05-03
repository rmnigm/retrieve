"""Encode each work in `item_id_map.json` into a frozen text embedding.

Inputs:
  - <data_dir>/item_id_map.json: work_id_str -> dense int (1-indexed; 0 = pad)
  - --catalog parquet (default ~/datasets/goodreads-ucsd/processed/goodreads_books.parquet)
    columns used: book_id, work_id, title, description (all strings)

Outputs:
  - <data_dir>/content/work_text_emb.pt: torch.float16 tensor of shape
      (num_items + 1, D_text), row 0 is the padding row (zeros)
  - <data_dir>/content/work_text_emb.meta.json: traceability of which encoder
      / template / truncation produced the .pt

For each work_id present in both the map and the catalog we pick the edition
with the longest non-empty description (ties -> first by book_id) and run the
sentence-transformer once. Works in the map but missing from the catalog are
left as zero rows and reported in the meta.json under `num_missing`.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import polars as pl
import torch
from loguru import logger
from tqdm import tqdm


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False),
    required=True,
    help="Directory containing item_id_map.json (e.g. data/goodreads/work-id).",
)
@click.option(
    "--catalog",
    type=click.Path(exists=True, dir_okay=False),
    default="~/datasets/goodreads-ucsd/processed/goodreads_books.parquet",
    show_default=True,
    help="Path to the Goodreads books parquet (one row per edition).",
)
@click.option(
    "--encoder",
    type=str,
    default="BAAI/bge-base-en-v1.5",
    show_default=True,
    help="HF sentence-transformer model id. D_text is read from the model.",
)
@click.option(
    "--text-template",
    type=str,
    default="{title}. {description}",
    show_default=True,
    help="Python str.format template; placeholders: {title}, {description}.",
)
@click.option(
    "--description-chars",
    type=int,
    default=1500,
    show_default=True,
    help="Char-level pre-truncate on the description before tokenization.",
)
@click.option("--batch-size", type=int, default=256, show_default=True)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--num-workers", type=int, default=8, show_default=True,
              help="DataLoader workers for parallel tokenization.")
@click.option(
    "--output-name",
    type=str,
    default="work_text_emb",
    show_default=True,
    help="Basename for the output .pt and .meta.json files.",
)
def main(
    data_dir: str,
    catalog: str,
    encoder: str,
    text_template: str,
    description_chars: int,
    batch_size: int,
    device: str,
    num_workers: int,
    output_name: str,
) -> None:
    data_path = Path(data_dir)
    map_path = data_path / "item_id_map.json"
    if not map_path.exists():
        raise click.UsageError(f"item_id_map.json not found at {map_path}")
    with open(map_path) as f:
        item_id_map: dict[str, int] = json.load(f)
    num_items = len(item_id_map)
    logger.info("Loaded item_id_map: {} works.", num_items)

    catalog_path = Path(catalog).expanduser()
    logger.info("Reading catalog: {}", catalog_path)
    df = pl.read_parquet(
        catalog_path, columns=["book_id", "work_id", "title", "description"]
    )
    logger.info("Catalog rows: {}", df.height)

    df = df.filter(pl.col("work_id").is_in(list(item_id_map.keys())))
    logger.info("Catalog rows after filter to map: {}", df.height)

    df = df.with_columns(
        pl.col("title").fill_null("").str.strip_chars().alias("title"),
        pl.col("description").fill_null("").str.strip_chars().alias("description"),
    ).with_columns(pl.col("description").str.len_chars().alias("desc_len"))

    df = df.sort(["work_id", "desc_len", "book_id"], descending=[False, True, False])
    df = df.unique(subset=["work_id"], keep="first", maintain_order=True)
    logger.info("Unique works after edition collapse: {}", df.height)

    work_ids: list[str] = df["work_id"].to_list()
    titles: list[str] = df["title"].to_list()
    descs: list[str] = df["description"].to_list()
    texts = [
        text_template.format(title=t, description=d[:description_chars])
        for t, d in zip(titles, descs, strict=True)
    ]

    from sentence_transformers import SentenceTransformer

    logger.info("Loading encoder: {}", encoder)
    model = SentenceTransformer(encoder, device=device)
    d_text = int(model.get_sentence_embedding_dimension())
    logger.info("Encoder dim: {}", d_text)

    text_emb = torch.zeros((num_items + 1, d_text), dtype=torch.float16)

    # Length-sort for batch packing efficiency, then unsort at the end.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    sorted_texts = [texts[i] for i in order]

    tokenizer = model.tokenizer
    max_seq_length = model.max_seq_length

    class _TextDS(torch.utils.data.Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    def _collate(batch):
        return tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_seq_length, return_tensors="pt",
        )

    loader = torch.utils.data.DataLoader(
        _TextDS(sorted_texts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=_collate,
        persistent_workers=False,
    )

    model.to(device).eval()
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    sorted_embs = torch.zeros((len(texts), d_text), dtype=torch.float16)

    pos = 0
    for batch in tqdm(loader, total=len(loader), desc="encode"):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device, dtype=autocast_dtype,
                           enabled=device == "cuda"),
        ):
            features = model(batch)
            embs = torch.nn.functional.normalize(
                features["sentence_embedding"], dim=-1,
            )
        n = embs.shape[0]
        sorted_embs[pos : pos + n] = embs.detach().to(dtype=torch.float16, device="cpu")
        pos += n

    # Unsort: row at original index `order[k]` should hold sorted_embs[k].
    order_t = torch.tensor(order, dtype=torch.long)
    text_emb_local = torch.empty_like(sorted_embs)
    text_emb_local[order_t] = sorted_embs

    encoded_ct = 0
    for i, work_id in enumerate(work_ids):
        text_emb[item_id_map[work_id]] = text_emb_local[i]
        encoded_ct += 1

    num_missing = num_items - encoded_ct

    out_dir = data_path / "content"
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"{output_name}.pt"
    meta_path = out_dir / f"{output_name}.meta.json"

    torch.save(text_emb, pt_path)
    meta = {
        "encoder": encoder,
        "dim": d_text,
        "num_items": num_items,
        "num_covered": encoded_ct,
        "num_missing": num_missing,
        "edition_pick_rule": "longest non-empty description, ties by book_id ascending",
        "text_template": text_template,
        "description_chars": description_chars,
        "shape": list(text_emb.shape),
        "dtype": "float16",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(
        "Wrote {} ({} covered, {} missing) and {}",
        pt_path,
        encoded_ct,
        num_missing,
        meta_path,
    )


if __name__ == "__main__":
    main()
