


#python main_vanilla_pinn.py
python grid_search.py --suffix=omg

#gs_dir="$(find . -maxdepth 1 -type d -name 'gridsearch__*__omg' | head -n 1)"
# if multiple match the regex, it sorts the matches and picks the last one, the newest one mased on the date
gs_dir="$(find . -maxdepth 1 -type d -name 'gridsearch__*__omg' | sort -r | head -n 1)"

echo "path = $gs_dir"
echo "realpath = $(realpath "$gs_dir")"

#cp -a "$src_dir" /path/to/destination/