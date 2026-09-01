English text follows below.

% Версия 2.1 (июнь 2022 года)
% Автор — Данил Фёдоровых (http://www.hse.ru/staff/df)

Руководство по использованию корпоративного стиля презентации НИУ ВШЭ в пакете Beamer.

1. Для использования шрифта HSE Sans в соответствии с брендбуком Высшей школы экономики, эти шрифты нужно установить в систему. Ссылка для скачивания: https://www.hse.ru/info/brandbook/#font

2. Для правильной компиляции документов нужно использовать движок XeLaTeX. Этот движок входит в современные дистрибутивы LaTeX (MikTeX, TeX Live), процедура его выбора зависит от конкретной программы-редактора (некоторые программы выберут XeLaTeX для компиляции документа автоматически, прочитав первую строку %!TEX TS-program = xelatex). Если вы раньше не работали с XeLaTeX, прочитайте краткую статью о его особенностях: http://ru.wikipedia.org/wiki/XeTeX . XeLaTeX — современный движок, который весьма прост в использовании; документы, сделанные для другого движка (например, pdfTeX), практически не нужно будет модифицировать для работы с новым движком.

3. Для создания презентаций рекомендуется модифицировать под свои нужды файл example-beamer-HSE.tex — в нем настроены все параметры для правильного отображения корпоративной темы. 

Что нового в версии 2.1
#  Добавлена возможность загрузки англоязычного логотипа

% Version 2.0 (January 2022)
% Danil Fedorovykh (http://www.hse.ru/en/staff/df)

HSE University official Beamer theme user manual.

1. To use font styles specified in the HSE University brand book, you should install the HSE Sans font. 

2. The document should be compiled in XeLaTeX. this engine comes with LaTeX distributions (MikTeX, TeX Live) and can be chosen in your .tex editor settings. (Some editors choose it automatically). XeLaTeX is modern and easy to use; only minimal amendments to your existing pdfTeX documents will be required. If you have never worked with XeLaTeX before, read the Wikipedia page: http://en.wikipedia.org/wiki/XeTeX.

3. In the file example-beamer-HSE-en.tex, all settings necessary for compilation are given in the preamble. It is better to modify this file for your own needs rather than integrate your existing Beamer files with the HSE theme.

What's new in version 2.1
#  HSE University logo in the English version is also in English