for l in $(seq 0 50 350); do
    h=$(($l + 400))
    python generate_window.py --ou_lo $l --ou_hi $h
done
