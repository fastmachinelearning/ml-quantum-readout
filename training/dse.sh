#/bin/bash 
################################# FLOATING-POINT #################################
# window range [0, 770]
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type mlp --exp 0
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type mlp --exp 1
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type mlp --exp 2
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type single --exp 0
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type single --exp 1
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type single --exp 2
# window range [200, 300]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type mlp --exp 0
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type mlp --exp 1
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type mlp --exp 2
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type single --exp 0
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type single --exp 1
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type single --exp 2
# window range [200, 400]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type mlp --exp 0
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type mlp --exp 1
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type mlp --exp 2
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type single --exp 0
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type single --exp 1
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type single --exp 2
# window range [200, 500]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type mlp --exp 0
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type mlp --exp 1
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type mlp --exp 2
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type single --exp 0
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type single --exp 1
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type single --exp 2
# window range [200, 600]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type mlp --exp 0
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type mlp --exp 1
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type mlp --exp 2
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type single --exp 0
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type single --exp 1
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type single --exp 2
# window range [200, 700]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type mlp --exp 0
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type mlp --exp 1
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type mlp --exp 2
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type single --exp 0
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type single --exp 1
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type single --exp 2

################################# QUANTIZATION #################################
# window range [0, 770]
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type mlp --exp 0 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type mlp --exp 1 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type mlp --exp 2 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type single --exp 0 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type single --exp 1 --quantize 
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type single --exp 2 --quantize 
# window range [200, 300]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type mlp --exp 0 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type mlp --exp 1 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type mlp --exp 2 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type single --exp 0 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type single --exp 1 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type single --exp 2 --quantize
# window range [200, 400]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type mlp --exp 0 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type mlp --exp 1 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type mlp --exp 2 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type single --exp 0 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type single --exp 1 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type single --exp 2 --quantize
# window range [200, 500]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type mlp --exp 0 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type mlp --exp 1 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type mlp --exp 2 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type single --exp 0 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type single --exp 1 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type single --exp 2 --quantize
# window range [200, 600]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type mlp --exp 0 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type mlp --exp 1 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type mlp --exp 2 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type single --exp 0 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type single --exp 1 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type single --exp 2 --quantize
# window range [200, 700]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type mlp --exp 0 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type mlp --exp 1 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type mlp --exp 2 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type single --exp 0 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type single --exp 1 --quantize
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type single --exp 2 --quantize

################################# QUANTIZATION & PRUNING #################################
# window range [0, 770]
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type mlp --exp 0 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type mlp --exp 1 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type mlp --exp 2 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type single --exp 0 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type single --exp 1 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 0 --window-end 770 --model-type single --exp 2 --quantize --prune
# window range [200, 300]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type mlp --exp 0 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type mlp --exp 1 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type mlp --exp 2 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type single --exp 0 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type single --exp 1 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 300 --model-type single --exp 2 --quantize --prune
# window range [200, 400]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type mlp --exp 0 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type mlp --exp 1 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type mlp --exp 2 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type single --exp 0 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type single --exp 1 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 400 --model-type single --exp 2 --quantize --prune
# window range [200, 500]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type mlp --exp 0 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type mlp --exp 1 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type mlp --exp 2 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type single --exp 0 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type single --exp 1 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 500 --model-type single --exp 2 --quantize --prune
# window range [200, 600]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type mlp --exp 0 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type mlp --exp 1 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type mlp --exp 2 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type single --exp 0 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type single --exp 1 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 600 --model-type single --exp 2 --quantize --prune
# window range [200, 700]
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type mlp --exp 0 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type mlp --exp 1 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type mlp --exp 2 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type single --exp 0 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type single --exp 1 --quantize --prune
python dse.py --data-dir ../data/malab_05012024/ --window-start 200 --window-end 700 --model-type single --exp 2 --quantize --prune
