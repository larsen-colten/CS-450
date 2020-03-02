from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
    # MYCT - machine_cycle_time_in_nanoseconds
    # MMIN - minimum main memory in kilobytes
    # MMAX - maximum main memory in kilobytes
    # CACH - cache memory in kilobytes
    # CHMIN - minimum channels in units
    # CHMAX - maximum channels in units
    # PRP - published relative performance
    # ERP - estimated relative performance from the original article
names = ['vender_name', 'model_name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
data_m = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/cpu-performance/machine.data", names = names, delimiter=',')
data_m.head()
obj_df = data_m.select_dtypes(include=['object', 'int64']).copy()
obj_df[obj_df.isnull().any(axis=1)]
clean_up_strings = { "vender_name": {"ibm": 1.01, "nas": 1.02, "honeywell": 1.03, "ncr": 1.04, "sperry": 1.05, "siemens": 1.06, 
                                      "amdahl": 1.07, "cdc": 1.08, "burroughs": 1.09, "dg": 1.011, "harris": 1.012, "hp": 1.013,
                                      "c.r.d": 1.014, "dec": 1.015, "magnuson": 1.016, "ipl": 1.017, "formation": 1.018, 
                                      "cambex": 1.019, "prime": 1.021, "perkin-elmer": 1.022, "nixdorf": 1.023, "gould": 1.024, 
                                      "bti": 1.025, "basf": 1.026, "wang": 1.027, "apollo": 1.028, "microdata": 1.029, 
                                      "sratus": 1.031, "four-phase": 1.032,"adviser": 1.033},
                      
                      "model_name": {"4381-1": 1.0101, "3081": 1.0102, "4321": 1.0103, "4341": 1.0104, "8140": 1.0105,          # ibm
                                      "3033:s": 1.0106, "3033:u": 1.0107, "3081:d": 1.0108, "3083:b": 1.0109,
                                      "3083:e": 1.01011, "370/125-2": 1.01012, "370/148": 1.01013, "370/158-3": 1.01014,
                                      "38/3": 1.01015, "38/4": 1.01016, "38/5": 1.01017, "38/7": 1.01018, "38/8": 1.01019,
                                      "4331-1": 1.01021, "4331-2":1.01022, "4331-11": 1.01023, "4341-1": 1.01024,
                                      "4341-10": 1.01025, "4341-11": 1.01026, "4341-12":1.01027, "4341-2":1.01028, 
                                      "4341-9": 1.01029, "4361-4": 1.01031, "4361-5": 1.01032, "4381-2": 1.01033,
                                      "8130-a": 1.01034, "8130-b": 1.01035,
                                      "as/3000": 1.0201, "as/3000-n": 1.0202, "as/5000": 1.0203, "as/5000-e": 1.0204,             # nas 
                                      "as/5000-n": 1.0205, "as/6130": 1.0206, "as/6150": 1.0207, "as/6620": 1.0208, 
                                      "as/6630": 1.0209, "as/6650": 1.02011, "as/7000": 1.02012, "as/7000-n":1.02013,
                                      "as/8040": 1.02014, "as/8050": 1.02015, "as/8060": 1.02016, "as/9000-dpc": 1.02017, 
                                      "as/9000-n": 1.02018, "as/9040": 1.02019, "as/9060": 1.02021,
                                      
                                      "dps:6/35": 1.0301, "dps:6/92": 1.0302, "dps:6/96": 1.0303, "dps:7/35": 1.0304,             # honeywell 
                                      "dps:7/45": 1.0305, "dps:7/55": 1.0306, "dps:7/65": 1.0307, "dps:8/20": 1.0308,
                                      "dps:8/44": 1.0309, "dps:8/49": 1.03011, "dps:8/50": 1.03012, "dps:8/52": 1.03013,
                                      "dps:8/62": 1.03014,
                                      "v8535:ii": 1.0401, "v8545:ii": 1.0402, "v8555:ii": 1.0403, "v8565:ii": 1.0404,             # ncr 
                                      "v8565:ii-e": 1.0405, "v8575:ii": 1.0406, "v8585:ii": 1.0407, "v8595:ii": 1.0408,
                                      "v8635": 1.0409, "v8650": 1.04011, "v8655": 1.04012, "v8665": 1.04013, "v8670": 1.04014,
                                      "1100/61-h1": 1.0501, "1100/81": 1.0502, "1100/82": 1.0503, "1100/83": 1.0504,              # sperry 
                                      "1100/84": 1.0505, "1100/93": 1.0506, "1100/94": 1.0507, "80/3": 1.0508, "80/4": 1.0509,
                                      "80/5": 1.05011, "80/6": 1.05012, "80/8": 1.05013, "90/80-model-3": 1.05014,
                                      "7.521": 1.0601, "7.531": 1.0602, "7.536": 1.0603, "7.541": 1.0604, "7.551": 1.0605,        # siemens
                                      "7.561": 1.0606, "7.865-2": 1.0607, "7.870-2": 1.0608, "7.872-2": 1.0609,  
                                      "7.875-2": 1.06011, "7.880-2": 1.06012, "7.881-2": 1.06013,
                                      "470v/7": 1.0701, "470v/7a": 1.0702, "470v/7b": 1.0703, "470v/7c": 1.0704,                  # amdahl 
                                      "470v/b": 1.0705, "580-5840": 1.0706, "580-5850": 1.0707, "580-5860": 1.0708,
                                      "580-5880": 1.0709, 
                                      "cyber:170/750": 1.0801, "cyber:170/760": 1.0802, "cyber:170/815": 1.0803,                  # cdc
                                      "cyber:170/825": 1.0804, "cyber:170/835": 1.0805, "cyber:170/845": 1.0806,
                                      "omega:480-i": 1.0807, "omega:480-ii": 1.0808, "omega:480-iii": 1.0809,
                                      "b1955": 1.0901, "b2900": 1.0902, "b2925": 1.0903, "b4955": 1.0904, "b5900": 1.0905,        # burroughs
                                      "b5920": 1.0906, "b6900": 1.0907, "b6925": 1.0908, 
                                      "eclipse:c/350": 1.01101, "eclipse:m/600": 1.01102, "eclipse:mv/10000": 1.01103,            # dg
                                      "eclipse:mv/4000": 1.01104, "eclipse:mv/6000": 1.01105, "eclipse:mv/8000": 1.01106, 
                                      "eclipse:mv/8000-ii": 1.01107, 
                                      "80": 1.01201, "100": 1.01202, "300": 1.01203, "500": 1.01204,                              # harris
                                      "600": 1.01205, "700": 1.01206, "800": 1.01207,
                                      "3000/30": 1.01301, "3000/40": 1.01302, "3000/44": 1.01303, "3000/48": 1.01304,             # hp
                                      "3000/64": 1.01305, "3000/88": 1.01306, "3000/iii": 1.01307, 
                                      "68/10-80": 1.01401, "universe:2203t": 1.01402, "universe:68": 1.01403,                     # c.r.d
                                      "universe:68/05": 1.01404, "universe:68/137": 1.01405, "universe:68/37": 1.01406,
                                      "decsys:10:1091": 1.01501, "decsys:20:2060": 1.01502, "microvax-1": 1.01503,                # dec
                                      "vax:11/730": 1.01504, "vax:11/750": 1.01505, "vax:11/780": 1.01506, 
                                      "m80/30": 1.01601, "m80/31": 1.01602, "m80/32": 1.01603,                                    # magnuson
                                      "m80/42": 1.01604, "m80/43": 1.01605, "m80/44": 1.01606, 
                                      "4436": 1.01701, "4443": 1.01702, "4445": 1.01703,                                          # ipl
                                      "4446": 1.01704, "4460": 1.01705, "4480": 1.01706,
                                      "f4000/100": 1.01801, "f4000/200": 1.01802, "f4000/200ap": 1.01803,                         # formation
                                      "f4000/300": 1.01804, "f4000/300ap": 1.01805,
                                      "1636-1": 1.01901, "1636-10": 1.01902, "1641-1": 1.01903,                                   # cambex
                                      "1641-11": 1.01904, "1651-1": 1.01905,
                                      "50-2250": 1.02101, "50-250-ii": 1.02102, "50-550-ii": 1.02103,                             # prime
                                      "50-750-ii": 1.02104, "50-850-ii": 1.02105,
                                      "3205": 1.02201, "3210": 1.02202, "3230": 1.02203,                                          # perkin-elmer
                                      
                                      "8890/30": 1.02301, "8890/50": 1.02302, "8890/70": 1.02303,                                 # nixdorf
                                      
                                      "concept:32/8750": 1.02401, "concept:32/8705": 1.02402, "concept:32/8780": 1.02403,         # gould
                                      
                                      "5000": 1.02501, "8000": 1.02502,                                                             # bti
                                      
                                      "7/65": 1.02601, "7/68": 1.02602,                                                     # basf
                                      
                                      "vs-100": 1.02701, "vs-90": 1.02702,                                                        # wang
                                      
                                      "dn320": 1.02801, "dn420": 1.02802,                                                         # apollo
                                      
                                      "seq.ms/3200": 1.02901,                                                                     # microdata
                                      
                                      "32": 1.03101,                                                                              # sratus
                                      
                                      "2000/260": 1.03201,                                                                        # four-phase
                                      
                                      "32/60": 1.03301}}                                                                           # adviser
obj_df.replace(clean_up_strings, inplace=True)
print(obj_df.head())
X = obj_df.iloc[:,0:9].to_numpy()
y = obj_df.iloc[:,9].to_numpy()

print(X)
print(y)
# define the keras model
model = Sequential()
model.add(Dense(9, input_dim=9))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
#compile
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset


model.fit(X, y, validation_split=.2, epochs=50, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))