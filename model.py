from wtforms import Form, FloatField, validators
from math import pi

class InputForm(Form):
    integrationtime = FloatField(
        label='integration time (days)', default=5.0,
        validators=[validators.NumberRange(0.5,360)])
    altitude = FloatField(
        label='altitude (km)', default=40.0,
        validators=[validators.NumberRange(20.0,50.0)])
    starttime = FloatField(
        label='start time [0-360 = 1st Jan-30th Dec]', default=0,
        validators=[validators.NumberRange(0,360)])
    latitude = FloatField(
        label='latitude [degrees from Greenwich]', default=0.0,
        validators=[validators.NumberRange(-90.0,90.0)])
    temperature = FloatField(
        label='temperature [Kelvin]', default=250.0,
        validators=[validators.NumberRange(0.0,400.0)])
    O3 = FloatField(
        label='ozone O3 [ppm]', default=7.0,
        validators=[validators.NumberRange(0.0,20.0)])
    O3P = FloatField(
        label='atomic oxygen O(3P) [ppq]', default=0.0,
        validators=[validators.NumberRange(0.0,0.0000005)])
    O1D = FloatField(
        label='excited state O(1D) [ppq]', default=0.0,
        validators=[validators.NumberRange(0.0,0.0000001)])
    NO = FloatField(
        label='NO [ppt]', default=0.015,
        validators=[validators.NumberRange(0.0,10000.0)])
    NO2 = FloatField(
        label='NO2 [ppb]', default=25.0,
        validators=[validators.NumberRange(0.0,30.0)])
    Cl = FloatField(
        label='Cl [ppq]', default=0.5,
        validators=[validators.NumberRange(0.0,100000.0)])
    ClO = FloatField(
        label='ClO [ppb]', default=0.35,
        validators=[validators.NumberRange(0.0,1000.0)])
    OH = FloatField(
        label='OH [ppt]', default=0.033,
        validators=[validators.NumberRange(0.0,100000.0)])
    HO2 = FloatField(
        label='HO2 [ppt]', default=1.85,
        validators=[validators.NumberRange(0.0,100000.0)])
    H = FloatField(
        label='H [ppq]', default=0.000007,
        validators=[validators.NumberRange(0.0,100000000.0)])
    Br = FloatField(
        label='Br [ppq]', default=3.33,
        validators=[validators.NumberRange(0.0,10000.0)])
    BrO = FloatField(
        label='BrO [ppt]', default=69.0,
        validators=[validators.NumberRange(0.0,1000.0)])
    CH4 = FloatField(
        label='CH4 [ppb]', default=410.0,
        validators=[validators.NumberRange(0.0,3000.0)])
    H2O = FloatField(
        label='specific humidity H2O [ppm]', default=4.0,
        validators=[validators.NumberRange(0.0,20.0)])
    N2O = FloatField(
        label='N2O [ppb]', default=50.0,
        validators=[validators.NumberRange(0.0,500.0)])
    HNO3 = FloatField(
        label='HNO3 [ppb]', default=0.82,
        validators=[validators.NumberRange(0.0,50.0)])
    HCl = FloatField(
        label='HCl [ppb]', default=4.3,
        validators=[validators.NumberRange(0.0,30.0)])
    ClONO2 = FloatField(
        label='ClONO2 [ppb]', default=0.6,
        validators=[validators.NumberRange(0.0,15.0)])
    HOCl = FloatField(
        label='HOCl [ppb]', default=0.33,
        validators=[validators.NumberRange(0.0,15.0)])
    ClOOCl = FloatField(
        label='ClOOCl [ppb]', default=0.6,
        validators=[validators.NumberRange(0.0,1000.0)])
    # https://ntrs.nasa.gov/citations/19890005228
    ClOO = FloatField(
        label='ClOO [ppt]', default=50.0,
        validators=[validators.NumberRange(0.0,500.0)])
    BrCl = FloatField(
        label='BrCl [ppt]', default=11.0,
        validators=[validators.NumberRange(0.0,100.0)])
    OClO = FloatField(
        label='OClO [ppt]', default=50.0,
        validators=[validators.NumberRange(0.0,500.0)])

class InputForm_mr(Form):
    watervapour = FloatField(
        label='water vapour vmr (ppmv)', default=5.0,
        validators=[validators.NumberRange(0.0,200.0)])
    # altitude = FloatField(
    #     label='altitude (km)', default=30.0,
    #     validators=[validators.NumberRange(12.0,50.0)])
    # starttime = FloatField(
    #     label='start time [0-360 = 1st Jan-30th Dec]', default=0,
    #     validators=[validators.NumberRange(0,360)])
    # latitude = FloatField(
    #     label='latitude [degrees]', default=0.0,
    #     validators=[validators.NumberRange(-90.0,90.0)])
    # temperature = FloatField(
    #     label='temperature [Kelvin]', default=210.0,
    #     validators=[validators.NumberRange(0.0,400.0)])

### here add heterogenous chemistry on/off button
