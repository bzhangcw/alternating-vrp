wdir="dataset/solomon-100-original"
hyperparams="--sigma 1.8 --tsig 1.5 --rho0 1"
output="output-$(date +"%Y%M%d%H%M")"

mkdir -p $output
if [ -f cmd.sh ]; then
    rm cmd.sh
fi

# vehicles, customers
scales=(
  "4,25"
  "8,50"
)

for size in $scales; do
  IFS=',' read nv nc <<< "${size}"

  for ff in $(ls $wdir/*.txt); do
    fname=$(basename -s .txt $ff)
    cmd="python main.py --fp $ff ${hyperparams} --n_vehicles $nv --n_customers $nc --output $output/$fname.$nv-$nc.json &> $output/$fname.$nv-$nc.log"
    echo $cmd >> cmd.sh
  done
done

echo "command saving to cmd.sh"
echo "create ./$output for saving solutions..."

