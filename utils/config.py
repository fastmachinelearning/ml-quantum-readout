precision = 'ap_fixed<16,6>'
reusefactor = 120000

config ={}

config['Model'] = {}
config['Model']['bn'] = {}
config['Model']['linear1'] = {}
config['Model']['linear2'] = {}
config['Model']['bn']['Precision'] = {}
config['Model']['linear1']['Precision'] = {}
config['Model']['linear2']['Precision'] = {}
# weight
config['Model']['bn']['Precision']['scale'] = precision
config['Model']['linear1']['Precision']['weight'] = precision
config['Model']['linear2']['Precision']['weight'] = precision
# bias
config['Model']['bn']['Precision']['bias'] = precision
config['Model']['linear1']['Precision']['bias'] = precision
config['Model']['linear2']['Precision']['bias'] = precision
# result
config['Model']['bn']['Precision']['result'] = precision
config['Model']['linear1']['Precision']['result'] = precision
config['Model']['linear2']['Precision']['result'] = precision
# reusefactor
config['Model']['bn']['ReuseFactor'] = reusefactor
config['Model']['linear1']['ReuseFactor'] = reusefactor
config['Model']['linear2']['ReuseFactor'] = reusefactor
## activation
config['Model']['relu1'] = {}
config['Model']['relu2'] = {}
config['Model']['relu1']['Precision'] = {}
config['Model']['relu2']['Precision'] = {}
config['Model']['relu1']['Precision'] = precision
config['Model']['relu2']['Precision'] = precision
config['Model']['relu1']['ReuseFactor'] = reusefactor
config['Model']['relu2']['ReuseFactor'] = reusefactor
