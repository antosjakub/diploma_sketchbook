


for combo in op_56_{1,2,3,4,5}_64; do
  IFS='_' read -r type n d ft <<< "$combo"
  echo "$type" "$n" "$d" "$ft"
  julia nd_2_heat.jl "$type" "$n" "$d" "$ft"
done
