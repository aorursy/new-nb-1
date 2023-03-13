# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter

import seaborn as sns

import datetime
from datetime import timedelta  

import math

#formatting functions for charts
def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fM' % (x * 1e-6)

#formatting functions for charts
def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.1fT' % (x * 1e-3)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
'''
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
display(train.head())

test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
display(test.head())

submission = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
display(submission.head())
'''
train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
train['Date'] = train['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))

train['NewFatalities'] = train['Fatalities'].diff(1)/1
train['NewCases'] = train['ConfirmedCases'].diff(1)/1

#display(train.head(5))

print("Count of Country_Region: ", train['Country_Region'].nunique())
print("Countries with Province/State: ", train[train['Province_State'].isna()==False]['Country_Region'].unique())
print("Date range: ", min(train['Date']), " - ", max(train['Date']))

display(train.head())
#create two new columns: 'Region' and 'State' to bring European countries into a single region. 
#All other Country_Regions with Province_State are also captured in these two columns Region, State

#EU data from https://www.google.com/publicdata/explore?ds=mo4pjipima872_&met_y=population&idim=country_group:eu&hl=en&dl=en#!ctype=l&strail=false&bcs=d&nselm=h&met_y=population&scale_y=lin&ind_y=false&rdim=country_group&idim=country_group:eu&idim=country:ea18:at:be:bg&ifdim=country_group&hl=en_US&dl=en&ind=false
#US States from https://worldpopulationreview.com/states/#statesTable

Europe=[
    'Albania',
    'Armenia',
    'Azerbaijan',
    'Austria', 
    'Belgium', 
    'Bulgaria',
    'Croatia',
    'Cyprus',
    'Czechia',
    'Denmark',
    'Estonia',
    'Finland', 
    'France', 
    'Germany', 
    'Greece', 
    'Hungary',
    'Iceland', 
    'Ireland', 
    'Italy',
    'Latvia',
    'Lichtenstein',
    'Lithuania',
    'Luxembourg',
    'Malta',
    'Montenegro',
    'Netherlands',
    'North Macedonia',
    'Norway', 
    'Poland',
    'Portugal',
    'Romania',
    'Slovakia',
    'Slovenia',
    'Spain', 
    'Sweden', 
    'Switzerland', 
    'United Kingdom'
]

train['Province_State'].fillna('',inplace=True)

train['State'] = train['Province_State']
train['Region'] = train['Country_Region']

train.loc[train['Country_Region'].isin(Europe),'Region']='EU'
train.loc[train['Country_Region'].isin(Europe),'State']=train.loc[train['Country_Region'].isin(Europe),'Country_Region']

#census populations
#add entries to this table in order to run simulations
Population = {
    'China-': 1386e6,
    'US-': 327e6,
    'EU-': 512e6 + (10+9+5+3+3+2+0.5+0.4)*1e6,

    'US-California':39937489,
    'US-Texas':29472295,
    'US-Florida':21992985,
    'US-New York':19440469,
    'US-Pennsylvania':12820878,
    'US-Illinois':12659682,
    'US-Ohio':11747694,
    'US-Georgia':10736059,
    'US-North Carolina':10611862,
    'US-Michigan':10045029,
    'US-New Jersey':8936574,
    'US-Virginia':8626207,
    'US-Washington':7797095,
    'US-Arizona':7378494,
    'US-Massachusetts':6976597,
    'US-Tennessee':6897576,
    'US-Indiana':6745354,
    'US-Missouri':6169270,
    'US-Maryland':6083116,
    'US-Wisconsin':5851754,
    'US-Colorado':5845526,
    'US-Minnesota':5700671,
    'US-South Carolina':5210095,
    'US-Alabama':4908621,
    'US-Louisiana':4645184,
    'US-Kentucky':4499692,
    'US-Oregon':4301089,
    'US-Oklahoma':3954821,
    'US-Connecticut':3563077,
    'US-Utah':3282115,
    'US-Iowa':3179849,
    'US-Nevada':3139658,
    'US-Arkansas':3038999,
    'US-Puerto Rico':3032165,
    'US-Mississippi':2989260,
    'US-Kansas':2910357,
    'US-New Mexico':2096640,
    'US-Nebraska':1952570,
    'US-Idaho':1826156,
    'US-West Virginia':1778070,
    'US-Hawaii':1412687,
    'US-New Hampshire':1371246,
    'US-Maine':1345790,
    'US-Montana':1086759,
    'US-Rhode Island':1056161,
    'US-Delaware':982895,
    'US-South Dakota':903027,
    'US-North Dakota':761723,
    'US-Alaska':734002,
    'US-District of Columbia':720687,
    'US-Vermont':628061,
    'US-Wyoming':567025,
    
    'EU-Vatican City':801,
    'EU-United Kingdom':67886011,
    'EU-Ukraine':43733762,
    'EU-Turkey':84339067,
    'EU-Switzerland':8654622,
    'EU-Sweden':10099265,
    'EU-Spain':46754778,
    'EU-Slovenia':2078938,
    'EU-Slovakia':5459642,
    'EU-Serbia':8737371,
    'EU-San Marino':33931,
    'EU-Russia':145934462,
    'EU-Romania':19237691,
    'EU-Portugal':10196709,
    'EU-Poland':37846611,
    'EU-Norway':5421241,
    'EU-Netherlands':17134872,
    'EU-Montenegro':628066,
    'EU-Monaco':39242,
    'EU-Moldova':4033963,
    'EU-Malta':441543,
    'EU-Luxembourg':625978,
    'EU-Lithuania':2722289,
    'EU-Liechtenstein':38128,
    'EU-Latvia':1886198,
    'EU-Kazakhstan':18776707,
    'EU-Italy':60461826,
    'EU-Ireland':4937786,
    'EU-Iceland':341243,
    'EU-Hungary':9660351,
    'EU-Greece':10423054,
    'EU-Germany':83783942,
    'EU-Georgia':3989167,
    'EU-France':65273511,
    'EU-Finland':5540720,
    'EU-Faroe Islands':48863,
    'EU-Estonia':1326535,
    'EU-Denmark':5792202,
    'EU-Czech Republic':10708981,
    'EU-Cyprus':1207359,
    'EU-Croatia':4105267,
    'EU-Bulgaria':6948445,
    'EU-Bosnia and Herzegovina':3280819,
    'EU-Belgium':11589623,
    'EU-Belarus':9449323,
    'EU-Azerbaijan':10139177,
    'EU-Austria':9006398,
    'EU-Armenia':2963243,
    'EU-Andorra':77265,
    'EU-Albania':2877797,
    
    'China-Hubei':59e6, #wuhan=11, hubei=59 59e6
    'Singapore-': 5.6e6, #not enough data to calibrate
    'Japan-': 127e6
}



#######################################################
# SIR model with INTERVENTION
#------------------------------------------------------
# params:
#
# x          : array of number of days since inception (not used except to size output); in the calibration below, inception starts on the first day reported fatalities reach a CUTOFF threshold
# i0         : initial percentage of infected population, for 1 per million: i0 = 1e-6
# beta       : initial daily rate of transmission by infected people to susceptible people, for R0=2.7 and gamma=1/21: beta=R0*gamma=2.7/21 
# gamma      : daily rate of recovery or death of infected people, for a 21 day speed of recovery or death: gamma = 1/21
# death_rate : daily death rate of infected people (assuming 1% of infected people die about 3 weeks after infection: death_rate=0.01/21)
#
# intervention_day : number of days after inception for intervention to start to reduce the initial transmission rate (beta)
# intervention_lag : number of days it takes for intervention to reach full effect (linear interp)
# intervention_effect : percentage reduction of initial transmission  rate, 0.25 for 25% reduction of initial beta after full intervention takes effect
########################################################

#-------------------------------------------------------
#the number returned by this function will be multiplied with the initial beta in order to estimate the transmission rate each day of the simulation
def intervention(day, day0, lag=5, effect=0.25):
    if day>day0+lag:
        return 1.0 - effect
    if day>day0:
        return 1.0 - effect * (day-day0)/lag
    return 1.0

'''
days = np.arange(300)
effects = np.zeros(300)
for d in days:
    effects[d] = intervention(d, 200, 3, 0.75)
plt.plot(days, effects)
plt.show()
'''

#-------------------------------------------------------
# basic daily integration of a classic SIR model with a time-variable beta parameter=beta*intervention(day)
# the function returns a numpy matrix, with a row per day and the following columns (cumulative results since day of inception)
cS  = 0  #Susceptible people
cI  = 1  #Infected people
cR  = 2  #Recovered people
cF  = 3  #Fatalities
cP  = 4  #Positive cases (recovered people are not included)

def SIR4(x, population, p0, f0, mixing, mixing_s, beta, phi, q, gamma, death_rate, intervention_day, intervention_lag, intervention_effect, detection_rate, detection_rate2):
    
    y = np.zeros((x.size,5))

    for i in range(0,x.size):
        
        if i==0:
            #initial conditions
            positives = p0
            infected = p0 / detection_rate
            fatalities = f0 #this does not change the dynamics, it only translates the fatalities curve up or down
            recovered = fatalities / death_rate
            susceptible = population - infected - fatalities - recovered
          
        else:
            #compute daily variations           
            rate = beta * intervention(i, intervention_day, intervention_lag, intervention_effect)
            rate = rate * ((1-phi)*math.exp(-q*i)+phi)   #beta decays from beta0 to phi*beta0 with half-life 1/q
            
            detect = max(0, min(1,detection_rate + detection_rate2 * i))  #assume a linear time-varying detection rate
                        
            d_fatalities = death_rate * infected
            d_recovered = (gamma - death_rate) * infected
            
            newlyinfected = rate * pow(susceptible/population, mixing_s) * pow(infected, mixing) #newly infected people, with a power law to have sub-exponential growth
            d_infected = newlyinfected - gamma * infected 
            d_susceptible = - newlyinfected
            d_positives = detect * newlyinfected
            
            #integrate and store in result array
            susceptible += d_susceptible
            positives += d_positives
            infected += d_infected
            recovered += d_recovered
            fatalities += d_fatalities
            
        y[i,cS] = susceptible
        y[i,cI] = infected
        y[i,cR] = recovered
        y[i,cF] = fatalities
        y[i,cP] = positives  #cumul of infected, does not come down on recovery. assuming all newly infected people are immediately detected
            
    return y


x = np.arange(300)

#plot number of fatalities 
#in a population on 1 million people, with one person initially infected, 
#assuming 3 weeks recovery rate, intial R0=5 and 1% death rate for infected people
population = 1e6  

#baseline: intervention has no effect in reducing initial transmission rate
y0 = SIR4(x, population=population, p0=1, f0=1, mixing=1, mixing_s=1, beta=5/21, phi=1, q=1, gamma=1.0/21, death_rate=0.01/21, 
         intervention_day = 0, intervention_lag = 3, intervention_effect = 0, detection_rate=1, detection_rate2=0)

# intervention starts on day 100 and results in 80% reduction of initial transmission rate 3 days later
y = SIR4(x, population=population, p0=1, f0=1, mixing=1, mixing_s=1, beta=5/21, phi=1, q=1, gamma=1.0/21, death_rate=0.01/21, 
         intervention_day = 60, intervention_lag = 3, intervention_effect = 0.30, detection_rate=1, detection_rate2=0)


fig,axs = plt.subplots(2,1, figsize=[8,8])

plt.subplot(211)
plt.title('Daily Fatalities')
plt.plot(x[1:], np.diff(y[:,cF]),'r-',label='daily fatalities (intervention)')
plt.plot(x[1:], np.diff(y0[:,cF]),'k-',label='daily fatalities (baseline)')
plt.legend()
#plt.yscale('log')

plt.subplot(212)
plt.title('Cumulative Fatalities')
plt.plot(x, y[:,cF],'r-',label='Fatalities (intervention)')
plt.plot(x, y0[:,cF],'k-',label='Fatalities (baseline)')
plt.legend()
#plt.yscale('log')

plt.show()


x = np.arange(200)

#plot number of fatalities 
#in a population on 1 million people, with one person initially infected, 
#assuming 3 weeks recovery rate, intial R0=5 and 1% death rate for infected people
population = 1e6  
p0=100
gamma = 1/14
R0 = 3
beta = R0 * gamma
death_rate = 0.01 * gamma

y25 = SIR4(x, population=population, p0=p0, f0=1, mixing=1, mixing_s=1, beta=2.5*gamma, phi=1, q=1, gamma=gamma, death_rate=death_rate, intervention_day = 0, intervention_lag = 0, intervention_effect = 0, detection_rate=1, detection_rate2=0)
y30 = SIR4(x, population=population, p0=p0, f0=1, mixing=1, mixing_s=1, beta=3.0*gamma, phi=1, q=1, gamma=gamma, death_rate=death_rate, intervention_day = 0, intervention_lag = 0, intervention_effect = 0, detection_rate=1, detection_rate2=0)
y35 = SIR4(x, population=population, p0=p0, f0=1, mixing=1, mixing_s=1, beta=3.5*gamma, phi=1, q=1, gamma=gamma, death_rate=death_rate, intervention_day = 0, intervention_lag = 0, intervention_effect = 0, detection_rate=1, detection_rate2=0)

fig,axs = plt.subplots(1,2, figsize=[12,6])
ax = plt.subplot(121)
plt.title('Daily Fatalities')
plt.plot(x[1:], np.diff(y25[:,cF]),'b-',label='R0=2.5')
plt.plot(x[1:], np.diff(y30[:,cF]),'r-',label='R0=3.0')
plt.plot(x[1:], np.diff(y35[:,cF]),'k-',label='R0=3.5')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
plt.xlabel('days since inception')
plt.legend()
plt.grid()

ax = plt.subplot(122)
plt.title('Cumulative Fatalities')
plt.plot(x, y25[:,cF],'b-',label='R0=2.5')
plt.plot(x, y30[:,cF],'r-',label='R0=3.0')
plt.plot(x, y35[:,cF],'k-',label='R0=3.5')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
plt.legend()
plt.xlabel('days since inception')
plt.grid()

caption = 'Daily and cumulative fatalities over time for a population of 1 million, 100 individuals are initially infected, \n14 days to recovery, 1% fatality rate\nR0 indicates the percentage of still susceptible people each infectious person will contaminate during their illness'
fig.text(.5, 1, caption, ha='center')

plt.show()
x = np.arange(150)

#plot number of fatalities 
#in a population on 1 million people, with one person initially infected, 
#assuming 3 weeks recovery rate, intial R0=5 and 1% death rate for infected people
population = 1e6  
p0=100
gamma = 1/14
R0 = 2
beta = R0 * gamma
death_rate = 0.01 * gamma

y1 = SIR4(x, population=population, p0=p0, f0=1, mixing=1, mixing_s=1, beta=beta, phi=1, q=1, gamma=gamma, death_rate=death_rate, intervention_day = 0, intervention_lag = 0, intervention_effect = 0, detection_rate=1, detection_rate2=0)
y95 = SIR4(x, population=population, p0=p0, f0=1, mixing=0.95, mixing_s=1, beta=beta, phi=1, q=1, gamma=gamma, death_rate=death_rate, intervention_day = 0, intervention_lag = 0, intervention_effect = 0, detection_rate=1, detection_rate2=0)
y90 = SIR4(x, population=population, p0=p0, f0=1, mixing=0.90, mixing_s=1, beta=beta, phi=1, q=1, gamma=gamma, death_rate=death_rate, intervention_day = 0, intervention_lag = 0, intervention_effect = 0, detection_rate=1, detection_rate2=0)

fig,axs = plt.subplots(1,2, figsize=[12,6])
ax = plt.subplot(121)
plt.title('Daily Fatalities')
plt.plot(x[1:], np.diff(y1[:,cF]),'b-',label='mixing=1.00')
plt.plot(x[1:], np.diff(y95[:,cF]),'r-',label='mixing=0.95')
plt.plot(x[1:], np.diff(y90[:,cF]),'k-',label='mixing=0.90')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
plt.yscale('log')
plt.xlabel('days since inception')
plt.legend()
plt.grid()

ax = plt.subplot(122)
plt.title('Cumulative Fatalities')
plt.plot(x, y1[:,cF],'b-',label='mixing=1.00')
plt.plot(x, y95[:,cF],'r-',label='mixing=0.95')
plt.plot(x, y90[:,cF],'k-',label='mixing=0.90')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
plt.yscale('log')
plt.legend()
plt.xlabel('days since inception')
plt.grid()

caption = 'Daily and cumulative fatalities over time for a population of 1 million, 100 individuals are initially infected, \n14 days to recovery, 1% fatality rate\nR0 indicates the percentage of still susceptible people each infectious person will contaminate during their illness'
fig.text(.5, 1, caption, ha='center')

plt.show()
x = np.arange(150)

#plot number of fatalities 
#in a population on 1 million people, with one person initially infected, 
#assuming 3 weeks recovery rate, intial R0=5 and 1% death rate for infected people
population = 1e6  
p0=100
gamma = 1/14
R0 = 2
beta = R0 * gamma
death_rate = 0.01 * gamma

y1 = SIR4(x, population=population, p0=p0, f0=1, mixing=1, mixing_s=1, beta=beta, phi=1, q=1, gamma=gamma, death_rate=death_rate, intervention_day = 0, intervention_lag = 0, intervention_effect = 0, detection_rate=1, detection_rate2=0)
y2 = SIR4(x, population=population, p0=p0, f0=1, mixing=1, mixing_s=2, beta=beta, phi=1, q=1, gamma=gamma, death_rate=death_rate, intervention_day = 0, intervention_lag = 0, intervention_effect = 0, detection_rate=1, detection_rate2=0)
y3 = SIR4(x, population=population, p0=p0, f0=1, mixing=1, mixing_s=3, beta=beta, phi=1, q=1, gamma=gamma, death_rate=death_rate, intervention_day = 0, intervention_lag = 0, intervention_effect = 0, detection_rate=1, detection_rate2=0)

fig,axs = plt.subplots(1,2, figsize=[12,6])
ax = plt.subplot(121)
plt.title('Daily Fatalities')
plt.plot(x[1:], np.diff(y1[:,cF]),'b-',label='mixing_s=1')
plt.plot(x[1:], np.diff(y2[:,cF]),'r-',label='mixing_s=2')
plt.plot(x[1:], np.diff(y3[:,cF]),'k-',label='mixing_s=3')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
plt.yscale('log')
plt.xlabel('days since inception')
plt.legend()
plt.grid()

ax = plt.subplot(122)
plt.title('Cumulative Fatalities')
plt.plot(x, y1[:,cF],'b-',label='mixing_s=1')
plt.plot(x, y2[:,cF],'r-',label='mixing_s=2')
plt.plot(x, y3[:,cF],'k-',label='mixing_s=3')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
plt.yscale('log')
plt.legend()
plt.xlabel('days since inception')
plt.grid()

caption = 'Daily and cumulative fatalities over time for a population of 1 million, 100 individuals are initially infected, \n14 days to recovery, 1% fatality rate\nR0 indicates the percentage of still susceptible people each infectious person will contaminate during their illness'
fig.text(.5, 1, caption, ha='center')

plt.show()
x = np.arange(200)

#plot number of fatalities 
#in a population on 1 million people, with one person initially infected, 
#assuming 3 weeks recovery rate, intial R0=5 and 1% death rate for infected people
population = 1e6  
p0=100
mixing=1
gamma = 1/14
R0 = 3
beta = R0 * gamma
phi=1
q=1
death_rate = 0.01 * gamma

y0 = SIR4(x, population=population, p0=p0, f0=1, mixing=1, mixing_s=1, beta=3.0*gamma, phi=1, q=1, gamma=gamma, death_rate=death_rate, intervention_day = 0, intervention_lag = 0, intervention_effect = 0, detection_rate=1, detection_rate2=0)
y1 = SIR4(x, population=population, p0=p0, f0=1, mixing=1, mixing_s=1, beta=3.0*gamma, phi=1, q=1, gamma=gamma, death_rate=death_rate, intervention_day = 20, intervention_lag = 1, intervention_effect = 0.3, detection_rate=1, detection_rate2=0)

fig,axs = plt.subplots(1,2, figsize=[12,6])
ax = plt.subplot(121)
plt.title('Daily Fatalities')
plt.plot(x[1:], np.diff(y0[:,cF]),'r-',label='no intervention')
plt.plot(x[1:], np.diff(y1[:,cF]),'k-',label='with intervention')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
plt.xlabel('days since inception')
plt.legend()
plt.grid()

ax = plt.subplot(122)
plt.title('Cumulative Fatalities')
plt.plot(x, y0[:,cF],'r-',label='no intervention')
plt.plot(x, y1[:,cF],'k-',label='with intervention')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
plt.legend()
plt.xlabel('days since inception')
plt.grid()

caption = 'Daily and cumulative fatalities over time for a population of 1 million, 100 individuals are initially infected, \n14 days to recovery, 1% fatality rate\ninitial R0=3.0 is the percentage of still susceptible people each infectious person will contaminate during their illness, before intervention \n intervention starts on day 20 and reaches full effect 1 day after, intervention reduces R0 by 30%'
fig.text(.5, 1, caption, ha='center')

plt.show()
#extract the data for the given region or state and prepare it for the calibration
def prep_data(data, region='US', state='New York', cutoff=1, truncate=0):
    
    c = data[data['Region']==region]
    if state != '':
        c = c[c['State']==state]
    
    c = c.groupby(['Date']).sum().reset_index()
    
    state = region + '-' + state
    c['State'] = state

    #find the first date when the fatalities cutoff was reached by this STATE, and keep only these days for calibration
    minDate = c[c['Fatalities']>cutoff]['Date'].min()
    
    s1 = c[c['Date']>minDate].copy()  #keep only the records after the given number of fatalities have been reached
    if truncate>0:
        s1 = s1[:truncate].copy()  #keep only the given number of days

    #calculate the number of days since the first day fatalities exceeded the cutoff
    s1['Days'] = (s1['Date'] - minDate) / np.timedelta64(1, 'D')

    return minDate, s1

def earlygrowth(data, region, state, cutoff):
    
    def growthmodel(x, **kwargs):  
        r = kwargs['r']
        a = kwargs['a']
        return np.exp(a * x) * r

    #https://arxiv.org/abs/1709.00973  
    #solution of sub-exponential growth of the form df/dt = r.f(t)^a
    def subgrowthmodel(x, a, b, r):
        if a==1:
            return b * np.exp(r*x)
        else:
            return (r*(1-a)*x + b**(1-a))**(1/(1-a))

    minDate, s1 = prep_data(data, region=region, state=state, cutoff=cutoff, truncate=0)
    population = Population[region + '-' + state]

    s1['NewFatalities'] = s1['Fatalities'].diff()
    s1['NewCases'] = s1['ConfirmedCases'].diff()

    x = s1['Days'].copy()

    fig, axs=  plt.subplots(1,3, figsize=(12,6))

    ax = plt.subplot(131)
    plt.plot(s1['Days'],s1['NewFatalities'], label='new fatalities')
    plt.plot(s1['Days'],s1['Fatalities'], label='fatalities')
    plt.plot(s1['Days'],s1['ConfirmedCases'], label='positives')
    plt.yscale('log')
    plt.legend()

    ax = plt.subplot(132)
    plt.plot(s1['Days'],s1['ConfirmedCases'], 'ko', label='positives')
    print("{}-{} positives:".format(region,state))
    z = s1['ConfirmedCases'].copy()
    for n in range(5, min(15,x.shape[0])):
        popt, pcov = curve_fit(subgrowthmodel,x[:n], z[:n]/population, bounds=((0.6,0,0),(1,1,1)), p0=(0.9,1e-9,2/14))
        print(popt)
        y1 = population * subgrowthmodel(x, a=popt[0], b=popt[1], r=popt[2])
        plt.plot(s1['Days'][:n],y1[:n], label='{} - {:.2f}'.format(n, popt[0]))
    plt.yscale('log')
    plt.legend()

    ax = plt.subplot(133)
    plt.plot(s1['Days'],s1['Fatalities'], 'ko', label='deaths')
    print("{}-{} fatalities:".format(region,state))
    z = s1['Fatalities'].copy()
    for n in range(5, min(10,x.shape[0])):
        popt, pcov = curve_fit(subgrowthmodel,x[:n], z[:n]/population, bounds=((0.6,0,0),(1,1,1)), p0=(0.9,1e-9,2/14))
        print(popt)
        y1 = population * subgrowthmodel(x, a=popt[0], b=popt[1], r=popt[2])
        plt.plot(s1['Days'][:n],y1[:n], label='{} - {:.2f}'.format(n, popt[0]))
    plt.yscale('log')
    plt.legend()

    plt.show()

region="EU"
state="Italy"
earlygrowth(train, region=region, state=state, cutoff=10)

###############################################
####### CALIBRATION TO reported Fatalities and Positive Test results
####### with intervention
###############################################

#Function called by scipy.curve_fit to calibrate the model parameters 
#This function calls the SIR model to simulate on the current guess parameters, and formats the results for use by curve_fit()
#note: SIR3() returns percentage of population, whereas reported data is absolute number of people, hence the need to use a total population number
#note: this function calibrates the final death rate (eg 1% of infected people eventually die, rather than the instantaneous death rate)
def SIR4_calib(x, p0, f0, mixing, mixing_s, beta, phi, q, gamma, death_rate,
                    intervention_day, intervention_lag, intervention_effect, detection_rate, detection_rate2, population, missing_data, scale_p, scale_f):
    
    y = SIR4(x, population=population, p0=p0, f0=f0, mixing=mixing, mixing_s=mixing_s, beta=beta, phi=phi, q=q, gamma=gamma, death_rate=death_rate*gamma,
            intervention_day = intervention_day, intervention_lag = intervention_lag, intervention_effect=intervention_effect, detection_rate=detection_rate, detection_rate2=detection_rate2)
    
    ret = np.append(y[:,cP] * scale_p, y[:,cF] * scale_f)    #positive counts are rescaled to same order of magnitude as death count for calibration algorithm - see prep_data()
    ret = np.where(missing_data, 0, ret)
    #ret = np.log(ret+1)
    return ret



#--------------------------
#This function calibrates and shows the results for one State
#data - DataFrame in the same format as train.csv but with Region and State columns added
#output - boolean, True to print results and charts
#cutoff - start simulation on the first day reported fatalities reach the cutoff level
#truncate - keep only this number of days of dato calibrate 
#forecast - forecast for this number of days
#
def calibrate(data, output=True, region='US', state='New York', force_mixing=1, force_mixing_s=1, force_gamma=1/7, interv_off=False, cutoff=1, truncate=0, forecast=365):

    if output:
        print('-----------------')
        print(region,'-',state)
        print('-----------------')
        print('')

    population = Population[region + '-' + state]

    #study the early growth to figure out whether it is exponential or sub-exponential
#    if output:
#        earlygrowth(data, region=region, state=state, cutoff=10)
    
    #prepare the calibration data
    #----------------------------
   
    #get the relevant data from the overall set
    
    minDate, s1 = prep_data(data, region=region, state=state, cutoff=cutoff, truncate=truncate)
    x = s1['Days']
    if output:
        print(cutoff, "Reported fatalities reached {} on {:%Y-%m-%d}".format(cutoff, minDate))
        print('')

    
    #calibrate on reported cumulative positives and fatalities, or on daily values
    
    calib_p = s1['ConfirmedCases'].copy()
    scale_p = calib_p.max()

    calib_f = s1['Fatalities'].copy()
    scale_f = calib_f.max()

    scale_p = 1 / scale_p
    scale_f = 1 / scale_f
    
    #calibrate on reported cumulative fatalities
    z = calib_p * scale_p
    z = z.append(calib_f * scale_f)
    missing_data = np.isnan(z) #record where we do not have data, for use by SEIR_calib to ignore these points during the calibration

    #z = np.log(1+z)
    z = np.nan_to_num(z)
        
    #calibrate the model
    #use a lambda to pass the population when running a simulation on guess parametes, but this is not a calibrated param
    #-------------------
    
    #calib_P0 = s1['ConfirmedCases'].iloc[0]
    #calib_F0 = s1['Fatalities'].iloc[0]
    calib_Mixing = force_mixing
    calib_MixingS = force_mixing_s
    calib_Phi = 1
    calib_Q = 1
    calib_Gamma = force_gamma
    calib_InterventionLag = 1
    
    #Bounds and initial guess for calibration algorithm scipy.curve_fit

    P0_min = 1e-6 * population
    P0_max = 10000e-6 * population
    Beta_min = 1.1 * calib_Gamma
    Beta_max = 5 * calib_Gamma
    DeathRate_min = 0.1e-2
    DeathRate_max = 10e-2
    InterventionDay_min = 1
    InterventionDay_max = 25
    InterventionEffect_min = 0
    InterventionEffect_max = 1
    DetectionRate_min = 1e-2  # detect 1% of infectious cases on day one
    DetectionRate_max = 1
    DetectionRate2_min = 0  #detection rate is constant in time
    DetectionRate2_max = 1/10  #rate of detection rate increase by day (capped at 100%). 1/10 would mean that it takes 10 days to reach 100% detection starting from 0% detection capacity

    initial_guess = [P0_min,
                     Beta_min,
                     DeathRate_min,
                     InterventionDay_max,  
                     InterventionEffect_min,
                     DetectionRate_min,
                     DetectionRate2_min]

    bounds = ((P0_min, Beta_min, DeathRate_min, InterventionDay_min, InterventionEffect_min, DetectionRate_min, DetectionRate2_min),
              (P0_max, Beta_max, DeathRate_max, InterventionDay_max, InterventionEffect_max, DetectionRate_max, DetectionRate2_max))
    


    def simpleSIR(x, p0, beta, death_rate, intervention_day, intervention_effect, detection_rate, detection_rate2):
        
        return SIR4_calib(x, p0=p0, f0=p0 * death_rate/detection_rate * gamma/(beta-gamma), 
                          mixing=calib_Mixing,mixing_s=calib_MixingS, beta=beta, phi=calib_Phi, q=calib_Q, 
                          gamma=calib_Gamma, death_rate=death_rate, 
                          intervention_day=intervention_day, intervention_lag=calib_InterventionLag, 
                          intervention_effect= 0 if interv_off else intervention_effect, 
                          detection_rate=detection_rate, detection_rate2=detection_rate2,population=population, missing_data=missing_data, scale_p=scale_p, scale_f=scale_f)        
    

    popt, pcov = curve_fit(simpleSIR,x, z, bounds=bounds, p0=initial_guess)

    
    calib_P0                 = popt[0]
    calib_Beta               = popt[1]
    calib_DeathRate          = popt[2]
    calib_InterventionDay    = popt[3]
    calib_InterventionEffect = 0 if interv_off else popt[4]
    calib_DetectionRate      = popt[5]
    calib_DetectionRate2     = popt[6]
    calib_I0                 = calib_P0 / calib_DetectionRate
    calib_F0                 = calib_P0 * calib_DeathRate / calib_DetectionRate * calib_Gamma/(calib_Beta-calib_Gamma)

    if output:
        print("SIR model fit")
        print("-------------")
        print("{} has a population of {:,.0f}".format(state, population))
        print("initiall positives are {:,.0f}".format(calib_P0))
        print("initial fatalities are {:,.0f}".format(calib_F0))
        print("current positives are {:,.0f}".format(s1['ConfirmedCases'].iloc[-1]))
        print("current fatalities are {:,.0f}".format(s1['Fatalities'].iloc[-1]))
        print("I0 = {:,.0f} per million, or {:,.0f} persons initially infected".format(calib_I0/population*1e6, calib_I0))
        print("MIXING = {:.2f}".format(calib_Mixing))
        print("MIXING_S = {:.2f}".format(calib_MixingS))
        print("BETA = {:.3f}".format(calib_Beta))
        print("PHI = {:.3f}".format(calib_Phi))
        print("Q = {:.3f}".format(1/calib_Q))
        print("GAMMA = {:.3f}, or {:.1f} days to recover".format(calib_Gamma, 1/calib_Gamma))
        print("DEATH RATE = {:.3%} infected people die".format(calib_DeathRate))
        print("R0 = {:.2f}".format(calib_Beta/calib_Gamma))
        print("Intervention Day = detected {:.0f} days after the cutoff, on {:%Y-%m-%d}".format(calib_InterventionDay, minDate+timedelta(days=calib_InterventionDay)))
        print("Intervention Lag = detected {:.0f} days for full intervention effect".format(calib_InterventionLag))
        print("Intervention Effect = detected {:.0%} reduction of initial transmission rate".format(calib_InterventionEffect))
        print("Detection Rate = {:.0%} infectious cases are reported as positives".format(calib_DetectionRate))
        print("Detection Rate Slope = {:.2%} daily increase of detection rate".format(calib_DetectionRate2))


    #compute model numbers for the calibration period
    #------------------------------------------------

    y = SIR4(x,population=population, p0=calib_P0, f0=calib_F0, mixing=calib_Mixing, mixing_s=calib_MixingS, beta=calib_Beta, phi=calib_Phi, q=calib_Q, gamma=calib_Gamma, death_rate=calib_Gamma*calib_DeathRate,
                         intervention_day = calib_InterventionDay, intervention_lag=calib_InterventionLag, intervention_effect = calib_InterventionEffect, detection_rate=calib_DetectionRate, detection_rate2=calib_DetectionRate2)

    s1['fit Fatalities (SIR)'] = y[:,cF]  #reported stats are about new cases, they do not seem to account for people having recovered
    s1['fit NewFatalities (SIR)'] = s1['fit Fatalities (SIR)'].diff()

    s1['fit Cases (SIR)'] = y[:,cP]  #reported positive stats are about new cases, they do not seem to account for people having recovered
    s1['fit NewCases (SIR)'] = s1['fit Cases (SIR)'].diff()

    #display(s1.sort_values(by='Date',ascending=False))

    #plot the model in comparison with calibrating data
    #--------------------------------------------------

    if output:
        fig,axs = plt.subplots(nrows=3, ncols=2,figsize=[16,16])

        plt.subplot(321)
        plt.title(state + ' Fatalities')
        plt.plot(x, s1['Fatalities'],'ko-',label='Actual')
        plt.plot(x, s1['fit Fatalities (SIR)'],'r-',label='SIR')
        plt.legend()
        plt.grid()
        plt.yscale('log')

        ax = plt.subplot(322)
        plt.title(state + ' New Fatalities')
        plt.plot(x, s1['NewFatalities'],'ko-',label='Actual')
        plt.plot(x, s1['fit NewFatalities (SIR)'],'r-',label='SIR')
        plt.legend()
        plt.grid()
        plt.yscale('log')

        ax = plt.subplot(323)
        plt.title(state + ' Confirmed Cases')
        plt.plot(x, s1['ConfirmedCases'],'ko-',label='Actual')
        plt.plot(x, s1['fit Cases (SIR)'],'b-',label='SIR')
        plt.legend()
        plt.grid()
        plt.yscale('log')
        #ax.yaxis.set_major_formatter(FuncFormatter(thousands))

        ax = plt.subplot(324)
        plt.title(state + ' New Cases')
        plt.plot(x, s1['NewCases'],'ko-',label='Actual')
        plt.plot(x, s1['fit NewCases (SIR)'],'b-',label='SIR')
        plt.legend()
        plt.grid()
        plt.yscale('log')
        #ax.yaxis.set_major_formatter(FuncFormatter(thousands))

    #long range forecast
    #-------------------

    xx = np.arange(forecast)
    y = SIR4(xx, population=population, p0=calib_P0, f0=calib_F0, mixing=calib_Mixing, mixing_s=calib_MixingS, beta=calib_Beta, phi=calib_Phi, q=calib_Q, gamma=calib_Gamma, death_rate=calib_Gamma*calib_DeathRate,
                          intervention_day = calib_InterventionDay, intervention_lag=calib_InterventionLag, intervention_effect = calib_InterventionEffect, detection_rate=calib_DetectionRate, detection_rate2=calib_DetectionRate2)


    idx = np.argmax( np.diff(y[:,cF])).item()   #peak daily fatalities
    max_dailyfatalities_day = minDate+timedelta(days=idx)
    max_dailyfatalities_rate = np.diff(y[:,cF])[idx]
    total_fatalities = y[-1,cF]

    if output:
        print("")
        print("SIR long range forecast")
        print("-----------------------")
        print("Daily New Fatalities would peak day {:,}, on {:%Y-%m-%d}, at {:,.0f} fatalities per day".format(idx,max_dailyfatalities_day, max_dailyfatalities_rate))
        print("Cumulative Fatalities would reach {:,.0f} after one year".format(total_fatalities))

        ax = plt.subplot(325)
        plt.title(state + ' Forecast')
        #plt.plot(xx, y[:,0],'g-',label='Susceptible')
        plt.plot(xx, y[:,cP],'r-',label='Positives')
        #plt.plot(xx, y[:,cR],'b-',label='Recovered')
        plt.plot(xx[1:], np.diff(y[:,cP]),'m-',label='Daily New Positives')
        ax.yaxis.set_major_formatter(FuncFormatter(millions))
        plt.legend()
        plt.grid()
        plt.yscale('log')

        ax = plt.subplot(326)
        plt.title(state + ' Forecast')
        lns2 = plt.plot(xx, y[:,cF],'c-',label='Fatalities (lhs)')
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))

        ax2 = ax.twinx() #instantiate second y axis, share same x axis
        lns3 = plt.plot(xx[1:], np.diff(y[:,cF]),'m-',label='Daily Fatalities (rhs)')
        ax2.yaxis.set_major_formatter(FuncFormatter(thousands))

        lns = lns2+lns3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        ax.grid()
        plt.yscale('log')

        plt.show()
    
    result = {}
    result['state'] = region + '-' + state
    result['curr fatalities'] = s1['Fatalities'].iloc[-1]
    result['population'] = population
    result['cutoff'] = minDate
    result['i0'] = calib_I0
    result['f0'] = calib_F0
    result['p0'] = calib_P0
    result['mixing'] = calib_Mixing
    result['phi'] = calib_Phi
    result['q'] = calib_Q
    result['gamma/d'] = 1/calib_Gamma
    result['R0'] = calib_Beta / calib_Gamma
    result['death rate'] = calib_DeathRate
    result['interv'] = calib_InterventionDay
    result['lag'] = calib_InterventionLag
    result['effect'] = calib_InterventionEffect
    result['detection'] = calib_DetectionRate
    result['detection2'] = calib_DetectionRate2
    result['peak day'] = max_dailyfatalities_day
    result['peak fatalities'] = max_dailyfatalities_rate
    result['cum fatalities'] = total_fatalities

    #print(result)
    
    return result, y
#try to calibrate perfect data

from datetime import datetime, timedelta
population=Population['US-New York']
mixing=1
mixing_s=1
gamma=1/14
beta=2*gamma
phi=1
q=1
death_rate=gamma*0.01
intervention_day = 0
intervention_lag=1
intervention_effect = 0
detection_rate=1
detection_rate2=0

p0 = 1
f0 = (death_rate/gamma) / detection_rate * gamma / (beta-gamma) * p0

n=200
x = np.arange(0,n)
y0 =  SIR4(x, population=population, p0=p0, f0=f0, mixing=mixing, mixing_s=mixing_s, beta=beta, phi=phi, q=q, gamma=gamma, death_rate=death_rate,
           intervention_day=intervention_day, intervention_lag=intervention_lag, intervention_effect=intervention_effect, 
           detection_rate=detection_rate, detection_rate2=detection_rate2)

test = pd.DataFrame(y0[:,cF], columns=['Fatalities'])
test['ConfirmedCases'] = y0[:,cP]
test['NewFatalities'] = test['Fatalities'].diff(1)/1
test['NewCases'] = test['ConfirmedCases'].diff(1)/1
test['Date'] = datetime(2020,1,1) + np.arange(n) * timedelta(days=1)
test['Region']='US'
test['State']='New York'

test['I'] = y0[:,cI]


display(test.tail())

#########
minDate, s1 = prep_data(test, region='US', state='New York', cutoff=10, truncate=0)

n=5
from scipy import stats
slopeI, interceptI, r_value, p_value, std_err = stats.linregress(x[:n], np.log(s1['I'][:n]))        #np.log(y[:n,cI]))
print("beta-gamma: %.3f true beta-gamma:  %.3f   I0: %f" % (slopeI, beta-gamma, np.exp(interceptI)))

slopeR, interceptR, r_value, p_value, std_err = stats.linregress(x[:n], np.log(s1['Fatalities'][:n]))
print("beta-gamma: %.3f true beta-gamma:  %.3f   intercept: %f" % (slopeR, beta-gamma, np.exp(interceptR)))


########
res, y =  calibrate(test, output=True, region='US', state='New York', force_mixing=1, force_mixing_s=1, force_gamma=gamma, interv_off=True, cutoff=1, truncate=0, forecast=365)




gamma = 1/7
region='US'
state ='California'
r,y = calibrate(train, output=True, region=region, state=state, force_mixing=1, force_mixing_s=1, force_gamma=gamma, interv_off=False, cutoff=25, truncate=0, forecast=365)


studies = [
    ['China',''],
    ['US',''],
    ['EU',''],
    ['EU','France'],
    ['EU', 'Italy'],
    ['EU', 'Spain'],
    ['EU', 'United Kingdom'],
    ['US', 'New York'],
    ['US', 'California']]


fig,ax = plt.subplots(figsize=[8,8])
plt.title('')
plt.legend()

gamma = 1/7
res = []
for s in studies:
    try:
        state = '{}-{}'.format(s[0],s[1])
        r,y = calibrate(train, output=False, region=s[0], state=s[1], force_mixing=1, force_mixing_s=1, force_gamma=gamma, interv_off=False, cutoff=1, truncate=0, forecast=365)
        res. append(r)
        plt.plot(np.diff(y[:,cF]),'k-',label=state)
    except:
        print(state, " failed to calibrate")

    
res2 = pd.DataFrame(res)

format_dict = {'population':'{:,.0f}', 
               'curr fatalities': '{:,.0f}',
               'cutoff':'{:%Y-%m-%d}',
               'i0/m': '{:.0f}',
               'gamma/d': '{:.0f}',
               'R0': '{:.1f}',
               'death rate': '{:.1%}',
               'interv': '{:.0f}',
               'lag': '{:.0f}',
               'effect': '{:.0%}',
               'detection': '{:.0%}',
               'detection2': '{:.1%}',
               'peak day':'{:%Y-%m-%d}',
               'peak fatalities': '{:,.0f}',
               'cum fatalities': '{:,.0f}',
              }

display(res2.style.format(format_dict).hide_index())



USStates = train[train['Region']=='US']['State'].unique()

res = []

fig,ax = plt.subplots(figsize=[8,8])
plt.title('')
plt.legend()

gamma = 1/7
for s in USStates:
    try:
        r,y = calibrate(train, output=False, region='US', state=s, force_mixing=1, force_mixing_s=1, force_gamma=gamma, interv_off=False, cutoff=1, truncate=0, forecast=365)
        res. append(r)
        plt.plot(np.diff(y[:,cF]),'k-',label=state)
    except:
        print(s, " failed to calibrate")
        
res2 = pd.DataFrame(res)

print('total fatalities could be {:,.0f}'.format(res2['cum fatalities'].sum()))

display(res2.sort_values(by='curr fatalities', ascending=False).style.format(format_dict).hide_index())

#display(train[train['Region']==region]['State'].unique())
gamma = 1/7
r,y = calibrate(train, output=True, region='EU', state='Italy', force_mixing=1, force_mixing_s=1, force_gamma=gamma, interv_off=False, cutoff=10, truncate=0, forecast=365)
r,y = calibrate(train, output=True, region='EU', state='France', force_mixing=1, force_mixing_s=1, force_gamma=gamma, interv_off=False, cutoff=10, truncate=0, forecast=365)
r,y = calibrate(train, output=True, region='EU', state='Spain', force_mixing=1, force_mixing_s=1, force_gamma=gamma, interv_off=False, cutoff=10, truncate=0, forecast=365)
r,y = calibrate(train, output=True, region='EU', state='United Kingdom', force_mixing=1, force_mixing_s=1, force_gamma=gamma, interv_off=False, cutoff=10, truncate=0, forecast=365)


gamma = 1/7
r,y = calibrate(train, output=True, region='China', state='Hubei', force_mixing=1, force_mixing_s=1, force_gamma=gamma, interv_off=False, cutoff=10, truncate=0, forecast=365)
gamma = 1/7
r,y = calibrate(train, output=True, region='US', state='New York', force_mixing=1, force_mixing_s=1, force_gamma=gamma, interv_off=False, cutoff=10, truncate=0, forecast=365)
r,y = calibrate(train, output=True, region='US', state='California', force_mixing=1, force_mixing_s=1, force_gamma=gamma, interv_off=False, cutoff=10, truncate=0, forecast=365)
#study the stability of the calibration on the China timeseries
#run the calibration as it would have been performed every day in the past on data then available, and plot the results



def backtest(data, region, state, gamma=1/7):

    minDate, s1 = prep_data(data, region=region, state=state, cutoff=1, truncate=0)
    start_run = 14  #start running the calibration 2 weeks after the report of one fatality
    end_run = s1.shape[0] - start_run #run the calibration every day until today
    print(start_run, end_run)

    #run the latest calibration as benchmark, and get the most accurate estimate for the timing of gov. intervention
    r,y = calibrate(data, output=True, region=region, state=state, force_mixing=1, force_mixing_s=1, force_gamma=gamma, interv_off=False, cutoff=10, truncate=0, forecast=250)
    interv = r['interv']
    effect = r['effect']

    res = []

    fig,ax = plt.subplots(figsize=[8,8])
    plt.title('')
    plt.yscale('log')


    for d in range(start_run, end_run):
        try:
            r,y = calibrate(data, output=False, region=region, state=state, force_mixing=1, force_mixing_s=1, force_gamma=gamma, interv_off=False, cutoff=10, truncate=d, forecast=250)
            r['truncated'] = d  #record the day for which this calibration would have been done
            res. append(r)

            if d>interv:  #we are running a calibration with data published after the intervention
                if r['effect'] < 0.5 * effect:  #has the effect been detected yet
                    style = 'g-'  #calibration after intervention but effect is not detected 
                else:
                    style = 'b-'  #calibration after intervention and effect is detected
            else:
                style = 'r-'  #calibration with data prior intervention

            plt.plot(np.diff(y[:,cF]),style,label=d)
        except:
            print(d, " failed to calibrate")

    plt.show()

    res2 = pd.DataFrame(res)

    fig, axs = plt.subplots(3,1,figsize=(12,6))

    ax = plt.subplot(311)
    plt.plot(res2['cum fatalities'],'ko-')
    plt.plot(res2['peak fatalities'], 'b*-')
    plt.yscale('log')
    plt.legend()

    ax = plt.subplot(312)
    plt.plot(res2['gamma/d'],'ko-')
    plt.plot(res2['death rate']*100, 'r*-')
    plt.legend()
    res2 = pd.DataFrame(res)

    ax = plt.subplot(313)
    plt.plot(res2['R0'], 'b*-')
    plt.legend()

    plt.show()

    format_dict2 = {'population':'{:,.0f}', 
                   'curr fatalities': '{:,.0f}',
                   'cutoff':'{:%Y-%m-%d}',
                   'i0/m': '{:.0f}',
                   'gamma/d': '{:.0f}',
                   'R0': '{:.1f}',
                   'death rate': '{:.1%}',
                   'interv': '{:.0f}',
                   'lag': '{:.0f}',
                   'effect': '{:.0%}',
                   'detection': '{:.0%}',
                   'detection2': '{:.1%}',
                   'peak day':'{:%Y-%m-%d}',
                   'peak fatalities': '{:,.0f}',
                   'cum fatalities': '{:,.0f}',
                  }

    display(res2.style.format(format_dict2).hide_index())        
        

region = 'China' 
state  = 'Hubei'
backtest(train, region, state, gamma=1/7)



def study(data, countries, truncate=0, interv_off=False, cutoff=50, force_mixing=1, force_mixing_s=1):

    fig, axs = plt.subplots(3,len(countries), figsize=(24,12))
    fig.autofmt_xdate()

    n1 = 200 #forecast range from cutoff day
    n2 = 50 #plot range

    res = []
    col=0
    for country in countries:

        region, state = country

        minDate, s1 = prep_data(data, region=region, state=state, cutoff=cutoff, truncate=truncate)
        x = minDate + np.arange(n1) * timedelta(days=1)
    
        for g in [4,6,8]:
            try:
                r, y = calibrate(train, output=False, region=region, state=state, force_mixing=force_mixing, force_mixing_s=force_mixing_s, force_gamma=1/g, interv_off=interv_off, cutoff=cutoff, truncate=truncate, forecast=n1)
                res.append(r)

                plt.subplot(axs[0][col])
                plt.plot(x[1:n2], np.diff(y[:n2,cP]),label='{}'.format(g))
                plt.plot(x[1:n2], np.diff(y[:n2,cF]),label='{}'.format(g))

                plt.subplot(axs[1][col])
                plt.plot(x[:n2], y[:n2,cP],label='{}'.format(g))
                plt.plot(x[:n2], y[:n2,cF],label='{}'.format(g))

                plt.subplot(axs[2][col])  #full range
                plt.plot(x, y[:,cP],label='{}'.format(g))
                plt.plot(x, y[:,cF],label='{}'.format(g))
                
            except:
                print('{}-{}: {} failed'.format(region, state, g))


        ax = plt.subplot(axs[0][col])
        plt.title(state)
        plt.plot(s1['Date'][:n2], s1['NewCases'][:n2], 'k+')
        plt.plot(s1['Date'][:n2], s1['NewFatalities'][:n2], 'r*')
        #plt.legend()
        plt.grid()
        plt.yscale('log')
        plt.ylim((1, 1e5))   # set the ylim to bottom, top
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%B-%d'))

        ax = plt.subplot(axs[1][col])
        plt.title(state)
        plt.plot(s1['Date'][:n2], s1['ConfirmedCases'][:n2], 'k+')
        plt.plot(s1['Date'][:n2], s1['Fatalities'][:n2], 'r*')
        #plt.legend()
        plt.grid()
        plt.yscale('log')
        plt.ylim((1, 1e6))   # set the ylim to bottom, top
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%B-%d'))            

        ax = plt.subplot(axs[2][col]) #full range
        plt.title(state)
        plt.plot(s1['Date'], s1['ConfirmedCases'], 'k+')
        plt.plot(s1['Date'], s1['Fatalities'], 'r*')
        #plt.legend()
        plt.grid()
        plt.yscale('log')
        plt.ylim((1, 2e7))   # set the ylim to bottom, top
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%B-%d'))            
        
        col=col + 1

    #handles, labels = ax.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center')
    
    format_dict2 = {'population':'{:,.0f}', 
                   'curr fatalities': '{:,.0f}',
                   'cutoff':'{:%Y-%m-%d}',
                   'i0': '{:,.0f}',
                   'gamma/d': '{:.0f}',
                   'R0': '{:.1f}',
                   'death rate': '{:.2%}',
                   'detection': '{:.1%}',
                   'interv': '{:.0f}',
                   'lag': '{:.0f}',
                   'effect': '{:.0%}',
                   'detection': '{:.0%}',
                   'detection2': '{:.1%}',
                   'peak day':'{:%Y-%m-%d}',
                   'peak fatalities': '{:,.0f}',
                   'cum fatalities': '{:,.0f}',
                  }

    res2 = pd.DataFrame(res)
    display(res2.style.format(format_dict2).hide_index())        
    return res2

            
truncate = 0
interv_off=False
cutoff=50
force_mixing=1
force_mixing_s=1

countries = [
    ('China','Hubei'),
    ('EU','Italy'),
    ('EU','France'),
    ('EU','Spain'),
    ('EU','United Kingdom'),
]
r = study(train, countries, truncate=truncate, interv_off=interv_off, cutoff=cutoff, force_mixing=force_mixing, force_mixing_s=force_mixing_s)
countries = [
    ('US','New York'),
    ('US','California'),
    ('US','New Jersey'),
    ('US','Florida'),
]
r = study(train, countries, truncate=truncate, interv_off=interv_off, cutoff=cutoff, force_mixing=force_mixing, force_mixing_s=force_mixing_s)

