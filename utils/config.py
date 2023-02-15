precision = 'ap_fixed<16,6>'
reusefactor = 1


def create_config(layers, precision='ap_fixed<16,6>', reusefactor=1):
    config = {}
    config['Model'] = {}
    config['Model']['Precision'] = precision
    config['Model']['ReuseFactor'] = reusefactor

    config['LayerName'] = {}

    for layer in layers:
        config['LayerName'][layer] = {}
        config['LayerName'][layer]['Trace'] = True
        config['LayerName'][layer]['Precision'] = {}
        config['LayerName'][layer]['ReuseFactor'] = reusefactor

    for layer in layers:
        if "bn" in layer:
            config['LayerName'][layer]['Precision']['scale'] = precision
        else:
            config['LayerName'][layer]['Precision']['weight'] = precision
        config['LayerName'][layer]['Precision']['bias'] = precision
        config['LayerName'][layer]['Precision']['result'] = precision
    return config


def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))