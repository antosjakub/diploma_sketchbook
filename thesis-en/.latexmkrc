# this is actually a file using perl programming language,
# do not worry bout the vscode error reporting

$aux_dir = 'tmp';
$pdf_mode = 4;

$pdflatex = 'pdflatex -interaction=nonstopmode %O %S';
# then build as:
# latexmk -pvc -pdf thesis.tex
