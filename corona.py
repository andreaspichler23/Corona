import matplotlib as mpl
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit 
import datetime
from pandas.plotting import register_matplotlib_converters
import matplotlib.dates as mdates
register_matplotlib_converters()

def func_exp(x, n0, tau):
    return n0 * 2**(x/tau)

# def func(x, n0, k):
#     g = (2/3)* 8800000
#     return g/(1+np.exp(-k*g*x)*(g/n0-1))

def func_logistic(x, x0, k):
    g = 0.65 * 8800000
    return g/(1+np.exp(-k*(x-x0)))

months = mdates.MonthLocator()
days = mdates.DayLocator(interval = 10)
months_fmt = mdates.DateFormatter('%Y-%m')
days_fmt = mdates.DateFormatter('%d')



base = datetime.date(2020, 2, 26)
cases = [2,2,4,5,10,10,18,29,41,55,79,99,131,182,246,302,428]
x_list = [i+1 for i in range(len(cases))]
x_list = np.asarray(x_list)
date_list = [base + datetime.timedelta(days=x) for x in range(len(cases))]

forecast_days = 90
x_list_long = [i+1 for i in range(forecast_days)]
x_list_long = np.asarray(x_list_long)
date_list_long = [base + datetime.timedelta(days=x) for x in range(forecast_days)]


# fig, ax = plt.subplots()
# # plt.yscale('log')
# plt.plot(date_list_long, func(x_list_long, 50, 1), 'r.')
# fig.autofmt_xdate()
# plt.show()

popt, pcov = curve_fit(func_logistic, x_list, cases, p0=[50,0.3])
popt_exp, pcov_exp = curve_fit(func_exp, x_list, cases)

tot_infections = func_logistic(x_list_long, *popt)
new_infections = [y - x for x,y in zip(tot_infections,tot_infections[1:])]
new_infections = [0] + new_infections
curr_infections = []

for i in range(forecast_days):
    if i < 14:
        curr_infections.append(tot_infections[i])
    else:
        curr_infections.append(tot_infections[i] - tot_infections[i-14])



print(popt)
# print(func(x_list_long, *popt))
# print('Days for double number of patients:', popt[1])
# print('Days for 10 times number of patients:', popt[1]*3.32)

print('estimated inflection day:', date_list_long[int(popt[0])] )


plt.plot(date_list,cases, 'b.')
plt.plot(date_list, func_logistic(x_list, *popt), 'r-')
plt.show()

fig, ax = plt.subplots()
ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(days)
ax.xaxis.set_major_formatter(months_fmt)
ax.xaxis.set_minor_formatter(days_fmt)

plt.yscale('log')
plt.plot(date_list_long, func_logistic(x_list_long, *popt), 'b.')
plt.plot(date_list_long, func_exp(x_list_long, *popt_exp), 'r.')
fig.autofmt_xdate()
plt.show()

# plt.yscale('log')
plt.plot(date_list_long, curr_infections, 'r-')
plt.plot(date_list_long, new_infections, 'b-')

fig.autofmt_xdate()
plt.show()
