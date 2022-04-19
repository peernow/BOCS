# License: GNU General Public License v3.0
# Author: Peer Nowack

from model import InputForm, InputForm_mr
from flask import Flask, render_template, request
from run import compute_chemistry
import sys

try:
    template_name = sys.argv[1]
except IndexError:
    template_name = 'about'

app = Flask(__name__)

@app.route('/')
def about():
    return render_template(template_name + '.html')

@app.route('/Modelrun/', methods=['GET', 'POST'])
def Modelrun():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        # try:
        #     template_name = sys.argv[1]
        # except IndexError:
        #     template_name = 'view2'
        # form_mr = InputForm_mr(request.form)
        # if request.method =='POST' and form_mr.validate():
       result = compute_chemistry(form.integrationtime.data, form.altitude.data,
                         form.starttime.data, form.latitude.data,
                        form.temperature.data,form.O3.data,
                        form.O3P.data,form.O1D.data,
                        form.NO.data,form.NO2.data,form.Cl.data,
                        form.ClO.data,form.OH.data,form.HO2.data,
                        form.H.data,form.Br.data,form.BrO.data,
                        form.CH4.data,form.H2O.data,form.N2O.data,
                        form.HNO3.data,form.HCl.data,form.ClONO2.data,
                        form.HOCl.data,form.ClOOCl.data,form.ClOO.data,
                        form.BrCl.data,form.OClO.data)
    else:
        result = None

    return render_template('Modelrun.html',
                           form=form, result=result)

@app.route('/Documentation/')
def Documentation():
    return render_template('Documentation.html')

@app.route('/Tutorial/')
def Tutorial():
    return render_template('Tutorial.html')

if __name__ == '__main__':
    app.run(debug=True)
