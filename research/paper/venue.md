---
title: venue
created: 2026-10-16
updated: 2026-10-16
type: summary
tags: [paper]
sources: [https://www.ecir2027.co.uk/call-for-reproducibility-papers, https://www.ecir2027.co.uk/call-for-full-papers, https://ecir2026.dryfta.com/calls/call-for-reproducibility-papers, https://ctan.org/pkg/llncs, https://resource-cms.springernature.com/springer-cms/rest/v1/content/19238648/data/v8, https://group.springernature.com/gp/group/ai/ai-guidance-for-our-researchers-and-communities]
---

# Venue: the ECIR 2027 replicability track

What the paper must satisfy, quoted from the
[ECIR 2027 call](https://www.ecir2027.co.uk/call-for-reproducibility-papers)
(read 2026-10-16). ECIR 2027 renamed the track: the page is still at
`call-for-reproducibility-papers`, but it is now titled "Call for
Replicability Papers". The review criteria are the 2026 list word for word,
except that "reproduction" became "replication" in one Impact question, so
[ECIR criteria](ecir-criteria.md) still applies as written. The paper itself
lives in [`tex/`](tex/README.md); what it needs before upload is in the
[submission checklist](submission-checklist.md).

## Dates and submission

| item | the call's words |
|---|---|
| abstract | "Abstract submission: 12 October 2026" (registered 2026-10-11: [abstract](abstract.md)) |
| paper | "Paper submission: 19 October 2026" |
| notification | "Paper notification: 7 December 2026" |
| conference | "21-25 March 2027 at Southampton Football Club, Southampton, England, UK", "a physical (in-person) conference" |
| system | "All papers should be submitted electronically through the Easychair submission system below": [easychair.org/conferences/?conf=ecir27](https://easychair.org/conferences/?conf=ecir27) |
| time zone | the replicability call states none. The full-paper call gives "11:59pm (AOE)" for its own deadlines; treat 19 October 23:59 AoE as the latest, not as a promise |

## Length and format

- "Replicability papers can be up to 12 pages in length plus additional
  pages for references. Appendices count toward the page limit."
- "Authors should consult Springer's author guidelines and use their
  proceedings templates, either for LaTeX or Word … The templates are
  available at https://bit.ly/springer-guidelines. Springer encourages
  authors to include ORCIDs in their papers."
- "All submissions must be written in English."
- The full-paper call adds "Please put appendices *before* the references
  for paper submission. Papers over the page limit will be desk rejected."
  The replicability call does not repeat it; the outline puts nothing in an
  appendix anyway ([outline](outline.md#page-budget)).
- **Template**: Springer LNCS, `llncs.cls` with `splncs04.bst`. Springer's
  download zip (the one `bit.ly/springer-guidelines` leads to) ships
  `llncs.cls` 2.24 (2024-01-29); CTAN ships 2.26 (2025-02-25: 2.25 fixes
  babel language switches, 2.26 ORCID hyphens). The `splncs04.bst` in both
  is byte-identical. We keep 2.26 in [`tex/`](tex/README.md). Since 2.23 the
  class has a `credits` environment and, in the class documentation's words,
  "the (mandatory) paragraph 'Disclosure of Interests', where any competing
  interests of the authors are to be declared".

## Anonymity

- "Replicability papers will be refereed through double-blind peer review,
  with an initial first stage review followed with a second stage of
  discussion led by a meta-reviewer."
- Desk rejection, among others, for "Submissions that are not submitted
  anonymously", "Formatting is not in line with the guidelines provided",
  "Addition of authors after the submission deadline".
- Preprints: "please do not publish your paper at arXiv and submit it to
  ECIR at the same time, some days before, or during the reviewing period of
  ECIR … Please do not cite your technical report and make some effort to
  avoid any issues that may harm the double-blindness of your submission."
- "Submissions from the same authors – i.e., repeatability (same team, same
  experimental setup) papers – of the reproduced experiments will not be
  accepted." We are not LiNR's or SilverTorch's authors.

What has to be anonymised for us (the GitHub repo, the Hub account, the
package name, acknowledgements) is planned in
[release and licenses §4](release-and-licenses.md#4-double-blind-review)
and listed in the [checklist](submission-checklist.md#anonymity).

## Artifacts

- "Authors are asked to upload their software artifacts either (i) to an
  anonymous repository that is linked to in the submission, such as an
  anonymous github [anonymous.4open.science], or (ii) to Easychair as part
  of the submission."
- The Availability criteria the reviewers apply: "Are the code and datasets
  used to reproduce the experiments available to the reviewers at the time
  of review? Is the shared material released in a permanent repository for
  easy access by researchers? Are the reproduced experiments well
  documented …? Are there discrepancies between what is described in the
  paper and what is available in the shared material? Is the shared
  material complete with everything you need to replicate the experiments
  exactly?"

## What the paper must say (content the call names)

"Emphasise your motivation for selecting the paper/papers, the process of
how results have been attempted to be replicated (successful or not), the
communication that was necessary to gather all information, the potential
difficulties encountered, and the result of the process. A successful
replication of the work is not a requirement." The call points to the ACM
"Artifact Review and Badging" terminology; in it, our study is a
*reproducibility* paper ("different team, different experimental setup"):
our data, hardware and LiNR code differ from the originals'.

## Statements: ethics, AI use, reproducibility

- **Ethics**: the call asks authors to adhere to the ACM policies on
  authorship, the ACM Code of Ethics, conflict of interest, plagiarism /
  misrepresentation / falsification, harassment, and "Springer Verlag's AI
  policy". No ethics statement section is required by the call.
- **AI use**: the call links Springer's AI policy
  (`link.springer.com/brands/springer/journal-policies#Artificial intelligence (AI)`),
  which serves a JavaScript challenge and could not be read. Springer
  Nature's wording as found in search (same policy, other Springer pages;
  **not read on the linked page itself**): "Large Language Models (LLMs),
  such as ChatGPT, do not currently satisfy our authorship criteria"; "Use
  of an LLM should be properly documented in the Methods section (and if a
  Methods section is not available, in a suitable alternative part) of the
  manuscript"; "The use of an LLM (or other AI-tool) for 'AI assisted copy
  editing' purposes does not need to be declared." Springer Nature's
  [AI guidance page](https://group.springernature.com/gp/group/ai/ai-guidance-for-our-researchers-and-communities)
  says it should be "declared in the Introduction or Acknowledgements". The
  call itself requires no AI-use statement at submission; whether and where
  ours goes is a [checklist](submission-checklist.md#statements) decision.
  (The full-paper call's "Use of AI tools" paragraph is addressed to
  reviewers, not authors.)
- **Reproducibility checklist**: none is required by the call.

## After acceptance

- "the corresponding author of each accepted paper … must complete and sign
  a Consent-to-Publish form. The corresponding author signing the copyright
  form should match the corresponding author marked on the paper. Once the
  paper has been submitted, changes relating to its authorship cannot be
  made."
- "Accepted papers will be published in the conference proceedings in the
  Springer Lecture Notes in Computer Science series."
- "Accepted papers will have to be presented at the conference by one of
  the authors in person – and at least one author for each accepted
  contribution will be required to register and attend."
