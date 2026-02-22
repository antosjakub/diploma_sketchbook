


for combo in {op,sparse}_64_{1,2,3,4}_{32,64}; do
  IFS='_' read -r type n d ft <<< "$combo"
  echo "$type" "$n" "$d" "$ft"
  julia nd_2_heat.jl "$type" "$n" "$d" "$ft"
done
