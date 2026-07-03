
# borrowing some files from the main thesis-en/
# (creating symlinks)


rm main.tex
ln -s ../../thesis-en/main.tex main.tex

rm .latexmkrc
ln -s ../../thesis-en/.latexmkrc .latexmkrc

rm bibliography.tex
ln -s ../../thesis-en/bibliography.tex bibliography.tex
