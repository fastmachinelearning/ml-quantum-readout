precision = 'ap_fixed<16,6>'
reusefactor = 1


def create_config(layers):
    config = {}
    config['Model'] = {}

    for layer in layers:
        config['Model'][layer] = {}
        config['Model'][layer]['Precision'] = {}
        config['Model'][layer]['ReuseFactor'] = reusefactor

    for layer in layers:
        config['Model'][layer]['Precision']['weight'] = precision
        config['Model'][layer]['Precision']['bias'] = precision
        config['Model'][layer]['Precision']['result'] = precision


def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))