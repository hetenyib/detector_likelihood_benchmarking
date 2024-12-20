# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from scipy.optimize import curve_fit


def det_from_p_fit(x, a, b):
    ''' 
    So called exponential saturation function. 
    Fits detector likelihood based on p value
    '''
    return a * (1 - np.exp(-b * x))
# det_from_p_fit


def det_from_p_fit_alt(x, alpha):
    ''' 
    So called exponential saturation function. 
    Fits detector likelihood based on p value
    '''
    return (1 - np.exp(-alpha * x)) / 2
# det_from_p_fit


def p_from_det_fit(x, a, b):
    '''
    Inverse of exponential saturation function
    used to find effective p for a given det likelihood input.
    Shot noise may result in average detector likelihoods over
    50%, so just setting the function to return coin flip probablity
    for detector likelihoods over 50%. 
    '''
    if x > .49999:
        return .5
    else:
        return - np.log(1 - x / a) / b
# End p_from_det_fit

def p_from_det_fit_alt(x, alpha):
    '''
    Inverse of exponential saturation function
    used to find effective p for a given det likelihood input.
    Shot noise may result in average detector likelihoods over
    50%, so just setting the function to return coin flip probablity
    for detector likelihoods over 50%. 
    '''
    if x > .49999:
        return .5
    else:
        return - np.log(1 - 2 * x) / alpha
# End p_from_det_fit


def log_er_per_round(sub_rounds, E_logical_round):
    return (1 - (1 - 2 * E_logical_round) ** sub_rounds) / 2
# End log_er_per_round


def get_effective_p(det_likelihood, size, sub_rounds):
    '''
    Given a code size, returns an eff_p value for given det_likelihood.
    '''
    size_fits = {11: {4: {'a': 0.489868338162893, 'b': 84.29021465727075, 'error': 0.0010729420030015484},
                    6: {'a': 0.489602058415151, 'b': 96.80850771230904, 'error': 0.001179653617754421},
                    8: {'a': 0.48942260613991073, 'b': 97.66101060708986, 'error': 0.0013070401266932646},
                    10: {'a': 0.48979338788645993, 'b': 97.52961323188663, 'error': 0.0013242564252249208},
                    12: {'a': 0.4906337514925499, 'b': 101.76945635789521, 'error': 0.0011494984591058628},
                    14: {'a': 0.4904152786132725, 'b': 102.15193689371563, 'error': 0.0011439033696739838},
                    16: {'a': 0.49061895834997205, 'b': 102.10053305132868, 'error': 0.0011632153987469569}},
                8: {4: {'a': 0.48778201729803966, 'b': 82.57027260472438, 'error': 0.0013224310265355587},
                    6: {'a': 0.4880634195451484, 'b': 94.58671008643509, 'error': 0.0015382013951770724},
                    8: {'a': 0.48734637428130206, 'b': 95.68692176569137, 'error': 0.0014629536251700443},
                    10: {'a': 0.48795914998898227, 'b': 95.48259607473668, 'error': 0.001511712770605529},
                    12: {'a': 0.48835952273609534, 'b': 99.82000489269143, 'error': 0.0012762632770800197},
                    14: {'a': 0.4885690105685817, 'b': 99.97377095166998, 'error': 0.0014515052016577518},
                    16: {'a': 0.4888384147185773, 'b': 99.8316882203107, 'error': 0.0013868995132738594}},
                5: {4: {'a': 0.48393110567366465, 'b': 80.0556537133641, 'error': 0.0012758205214947744},
                    6: {'a': 0.4840754970620359, 'b': 91.62354074600005, 'error': 0.0016473768473349922},
                    8: {'a': 0.4842153993453441, 'b': 92.50437777345411, 'error': 0.0016711868864538629},
                    10: {'a': 0.4851206617509904, 'b': 91.86277661606059, 'error': 0.0017222321492213987},
                    12: {'a': 0.4858704740307484, 'b': 95.95629145715017, 'error': 0.0016447053171961396},
                    14: {'a': 0.4854514024217009, 'b': 96.42806304685055, 'error': 0.0018128039980757507},
                    16: {'a': 0.48581571806463486, 'b': 96.28622666399853, 'error': 0.0016895195271243652}},
                3: {4: {'a': 0.48090252682442247, 'b': 76.47168911133066, 'error': 0.0023659646666229946},
                    6: {'a': 0.4831330866332295, 'b': 88.21013874702247, 'error': 0.002007095275289712},
                    8: {'a': 0.48367259688380965, 'b': 88.85561562448183, 'error': 0.0019174001352173166},
                    10: {'a': 0.4812781205704606, 'b': 88.49865980028815, 'error': 0.0020934421826121676},
                    12: {'a': 0.48461921885935905, 'b': 92.14225041005987, 'error': 0.0021541190221188136},
                    14: {'a': 0.4840371538835751, 'b': 92.75187088387166, 'error': 0.001962245122090337},
                    16: {'a': 0.48279021499408475, 'b': 92.36681454730325, 'error': 0.002095367748644974}},
                2: {4: {'a': 0.4843939922543833, 'b': 74.94877728381486, 'error': 0.0015887270270159846},
                    6: {'a': 0.47577262494658507, 'b': 88.8506210853406, 'error': 0.00262106407812677},
                    8: {'a': 0.4771174463193179, 'b': 88.90017229234375, 'error': 0.002577164890362932},
                    10: {'a': 0.4826044106045616, 'b': 86.9164719521806, 'error': 0.002031681751728932},
                    12: {'a': 0.47989922956983744, 'b': 91.77053558262381, 'error': 0.002713301747038579},
                    14: {'a': 0.4795638670304722, 'b': 92.25083829065554, 'error': 0.0022033099394005965},
                    16: {'a': 0.48311129186153795, 'b': 90.85606854271019, 'error': 0.001820078272510705}}}
                    
    a = size_fits[size][sub_rounds]['a']
    b = size_fits[size][sub_rounds]['b']
    if det_likelihood > a:
        return .5
    else:
        return p_from_det_fit(det_likelihood, a, b)
# End get_effective_p


def get_effective_p_alt(det_likelihood, size, sub_rounds):
    '''
    Given a code size, returns an eff_p value for given det_likelihood.
    '''
    size_fits = {11: {4: {'alpha': 80.63944608975558, 'error': 0.8995219131448383},
                    6: {'alpha': 92.47748646178404, 'error': 1.238356398464329},
                    8: {'alpha': 93.21502302377432, 'error': 1.2890383145673452},
                    10: {'alpha': 93.24501942038069, 'error': 1.249122898785052},
                    12: {'alpha': 97.6718509947859, 'error': 1.2476253014448002},
                    14: {'alpha': 97.94340922081734, 'error': 1.2811997003680486},
                    16: {'alpha': 97.98330256731322, 'error': 1.258435191709556}},
                8: {4: {'alpha': 78.26805505500197, 'error': 1.021904731032755},
                    6: {'alpha': 89.72598256595275, 'error': 1.3554964554974944},
                    8: {'alpha': 90.47422936876937, 'error': 1.4504434008419904},
                    10: {'alpha': 90.53236317065013, 'error': 1.3895418485160238},
                    12: {'alpha': 94.81992128212589, 'error': 1.4571813003326706},
                    14: {'alpha': 95.05401355177003, 'error': 1.4539719013446148},
                    16: {'alpha': 95.03544957376899, 'error': 1.415694708621493}},
                5: {4: {'alpha': 74.60229031403922, 'error': 1.198705472085363},
                    6: {'alpha': 85.35065393492812, 'error': 1.6171123043832465},
                    8: {'alpha': 86.22331276310229, 'error': 1.6393574497714953},
                    10: {'alpha': 85.98349204209642, 'error': 1.5428606461662342},
                    12: {'alpha': 90.11657070985106, 'error': 1.6124122498261428},
                    14: {'alpha': 90.38276818194488, 'error': 1.6810426085697958},
                    16: {'alpha': 90.40297733357065, 'error': 1.6323172358121467}},
                3: {4: {'alpha': 70.3300675501554, 'error': 1.2861674766988633},
                    6: {'alpha': 81.82683860649917, 'error': 1.5845704202469175},
                    8: {'alpha': 82.62782026633712, 'error': 1.5627643701488352},
                    10: {'alpha': 81.39076748040226, 'error': 1.7380583596306576},
                    12: {'alpha': 86.04014128693743, 'error': 1.6270289585512354},
                    14: {'alpha': 86.37816131715118, 'error': 1.6833760385454377},
                    16: {'alpha': 85.52452144392801, 'error': 1.7810872284282344}},
                2: {4: {'alpha': 70.04202755507677, 'error': 1.0160985715141395},
                    6: {'alpha': 79.62112002315071, 'error': 2.1686295929952117},
                    8: {'alpha': 80.17359402811842, 'error': 2.0781436215355042},
                    10: {'alpha': 80.43864545725607, 'error': 1.5741974467580977},
                    12: {'alpha': 83.82776881882532, 'error': 2.027062801564349},
                    14: {'alpha': 84.13993980943496, 'error': 2.0399641177462704},
                    16: {'alpha': 84.26103396762736, 'error': 1.6771373775021339}}}
    
    alpha = size_fits[size][sub_rounds]['alpha']

    return p_from_det_fit_alt(det_likelihood, alpha)
# End get_effective_p

def get_effective_p_no_truncs(det_likelihood, size, sub_rounds):
    '''
    Given a code size, returns an eff_p value for given det_likelihood.
    '''
    size_fits = {11: {4: {'a': 0.49008009707808714, 'b': 87.01054776194093, 'error': 0.0016184489125332378},
                    6: {'a': 0.4906455859602016, 'b': 94.81117839717437, 'error': 0.0013475610644227063},
                    8: {'a': 0.4926143820056988, 'b': 101.5464146315133, 'error': 0.0012306735028061865},
                    10: {'a': 0.4922355272001266, 'b': 102.67445332018319, 'error': 0.001109654122415973},
                    12: {'a': 0.4925850755815153, 'b': 102.53553499273345, 'error': 0.001206650171600124},
                    14: {'a': 0.4933353847391738, 'b': 104.80101174453041, 'error': 0.0010815894172826333},
                    16: {'a': 0.4931635861400433, 'b': 105.06890940620306, 'error': 0.001086293613581054}},
                8: {4: {'a': 0.4876326187580097, 'b': 85.06756191252047, 'error': 0.0014624270026413477},
                    6: {'a': 0.48916837449959705, 'b': 92.68282976136521, 'error': 0.0015235313270210247},
                    8: {'a': 0.49067189599962324, 'b': 99.38642084000924, 'error': 0.0015095504587065345},
                    10: {'a': 0.49109410760468614, 'b': 100.16851263844057, 'error': 0.0013752521370945193},
                    12: {'a': 0.49082393067003394, 'b': 100.26047275278958, 'error': 0.0014094204724257123},
                    14: {'a': 0.49162278340985904, 'b': 102.54798121468697, 'error': 0.0012674906290568765},
                    16: {'a': 0.4917175040423925, 'b': 102.80899714212299, 'error': 0.0012454671292792263}},
                5: {4: {'a': 0.48594432928523346, 'b': 80.31016800108269, 'error': 0.002542638291019741},
                    6: {'a': 0.48572057487826453, 'b': 89.39184346293618, 'error': 0.0016455478663093292},
                    8: {'a': 0.4886225829730899, 'b': 95.08376727440971, 'error': 0.0016876954016058273},
                    10: {'a': 0.48833958667161287, 'b': 96.41374237251104, 'error': 0.0018746499906214216},
                    12: {'a': 0.4887700357492443, 'b': 96.17827224269675,'error': 0.0016435508380096577},
                    14: {'a': 0.4896320245381289, 'b': 98.30646178990861, 'error': 0.0018942483182137143},
                    16: {'a': 0.48929025623267164, 'b': 98.95010098171699, 'error': 0.0016455035862877462}},
                3: {4: {'a': 0.48544029553256146, 'b': 82.32025157530816, 'error': 0.0018955504162968026},
                    6: {'a': 0.4854896556427473, 'b': 86.19570718686066, 'error': 0.0019093220192059614},
                    8: {'a': 0.487680609583918, 'b': 91.57334146485151, 'error': 0.002307515316892561},
                    10: {'a': 0.4874085024909857,'b': 92.49679461819865, 'error': 0.0018879153845386407},
                    12: {'a': 0.4877877633515109, 'b': 92.42183509907828, 'error': 0.0016885100022939678},
                    14: {'a': 0.48876617669830896, 'b': 94.14563442869382, 'error': 0.0020391129037734255},
                    16: {'a': 0.488473605783115, 'b': 94.15320944194983,'error': 0.0020209924666592482}},
                2: {4: {'a': 0.4769753297514337, 'b': 69.29128124659154, 'error': 0.0031715772763589957},
                    6: {'a': 0.4817824960296628, 'b': 84.41201103601658, 'error': 0.004456180591987633},
                    8: {'a': 0.483935257414947, 'b': 91.17591547532031, 'error': 0.0021992712370039547},
                    10: {'a': 0.48533363841288246, 'b': 91.49668554808547, 'error': 0.0028634890906266332},
                    12: {'a': 0.486285350718338, 'b': 90.79491150475904, 'error': 0.0025558418464919903},
                    14: {'a': 0.48445291012857605,'b': 93.89379809582982, 'error': 0.002634475363734204},
                    16: {'a': 0.48673363070833453,'b': 93.25761225388145, 'error': 0.002453000650245848}}}
    
             
    a = size_fits[size][sub_rounds]['a']
    b = size_fits[size][sub_rounds]['b']
    if det_likelihood > a:
        return .5
    else:
        return p_from_det_fit(det_likelihood, a, b)
# End get_effective_p


def get_effective_p_alt_no_truncs(det_likelihood, size, sub_rounds):
    '''
    Given a code size, returns an eff_p value for given det_likelihood.
    '''
    size_fits = {11: {4: {'alpha': 83.08999672521196, 'error': 1.220659699590126},
                    6: {'alpha': 90.81567245770678, 'error': 1.3386660692997676},
                    8: {'alpha': 98.2019554408518, 'error': 1.2385777535353089},
                    10: {'alpha': 99.12671347209844, 'error': 1.295283329181167},
                    12: {'alpha': 99.15056164822103, 'error': 1.2597886625033095},
                    14: {'alpha': 101.70472585694088, 'error': 1.1811617507059644},
                    16: {'alpha': 101.8853366654388, 'error': 1.2115865353351627}},
                8: {4: {'alpha': 80.28480262789945, 'error': 1.372781259400356},
                    6: {'alpha': 88.14339892557226, 'error': 1.4652715932953964},
                    8: {'alpha': 95.23030946930355, 'error': 1.4792288067567414},
                    10: {'alpha': 96.1760575710794, 'error': 1.4261188370469962},
                    12: {'alpha': 96.14255075203059, 'error': 1.4679878379542564},
                    14: {'alpha': 98.71919519104377, 'error': 1.3991811635890552},
                    16: {'alpha': 99.01618919774016, 'error': 1.389268807999961}},
                5: {4: {'alpha': 75.16114070775572, 'error': 1.4486332165082747},
                    6: {'alpha': 83.59585540021317, 'error': 1.7203011383665878},
                    8: {'alpha': 90.2011778466114, 'error': 1.619836924009326},
                    10: {'alpha': 91.34021068352186, 'error': 1.7194467950517047},
                    12: {'alpha': 91.3085485259103, 'error': 1.631511761539188},
                    14: {'alpha': 93.71867179002905, 'error': 1.6342526516187827},
                    16: {'alpha': 94.18938726725904, 'error': 1.6558266140686655}},
                3: {4: {'alpha': 76.8609321979554, 'error': 1.4994744008057659},
                    6: {'alpha': 80.50276621230985, 'error': 1.644566271298981},
                    8: {'alpha': 86.44885947102257, 'error': 1.686454392069561},
                    10: {'alpha': 87.2189448339049, 'error': 1.689272750617415},
                    12: {'alpha': 87.3110836673775, 'error': 1.6233753030497917},
                    14: {'alpha': 89.35899571444158, 'error': 1.623346730604818},
                    16: {'alpha': 89.24187706314612, 'error': 1.6518671537948135}},
                2: {4: {'alpha': 62.15574957034158, 'error': 1.5275890081431918},
                    6: {'alpha': 77.3490487995346, 'error': 2.1700833464348572},
                    8: {'alpha': 84.5149228295119, 'error': 2.0187728642991494},
                    10: {'alpha': 85.37999991134575, 'error': 1.9863989221371354},
                    12: {'alpha': 85.12931455366135, 'error': 1.8263953597373348},
                    14: {'alpha': 87.25557870013199, 'error': 2.1385019167466246},
                    16: {'alpha': 87.63700490899228, 'error': 1.861684377688212}}}
             
    alpha = size_fits[size][sub_rounds]['alpha']
    if det_likelihood > .499999:
        return .5
    else:
        return p_from_det_fit_alt(det_likelihood, alpha)
# End get_effective_p


def get_effective_p_single(D, alpha=118.7):
    return p_from_det_fit_alt(D, alpha)
