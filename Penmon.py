#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:42:59 2023
Compute Penman Monteith evapotranspiration for a given location.
@author: jimmy
"""
# import standard libs 
import math as ma 
from datetime import datetime, timedelta 
from warnings import warn 

#define some constants 
stefboltzman = 5.670374419e-8 # (stefan boltzman constant from google)
Cp = 1.013e-3 # MJ kg^-1 c^-1 - specifc heat at constant pressure 
lamda = 2.45 # MJ kg^-1 - latent heat of vapourisation 
epsilon = 0.622 # ratio of molecular weight of water versus air 
Gsolar = 0.0820 # solar constant 
alpha = 0.23 # canopy reflection coefficient of grass 

# functions 
def vapPres(T):
    return 0.6108*ma.exp((17.27*T)/(T+273.3))

class PETestimator():
    """Estimation potential evapotranspiration using the Penman Monteith equation
    and guidelines from Allen et al (1998). 

    Parameters
    ----------
    lat : float 
        Latitude in degrees             
    
    elev : float 
        Elevation of site (m)
        
    date : str, datetime.datetime
        day that observations are made. Can be string or python datetime object. 
             
    date_format : str, optional
        if date given as str then this describes the format. The default is '%Y-%m-%d'.
                     
    Tmean : float, optional
        Mean daily temperature (deg C). The default is None.If None then the 
        average of Tmax and Tmin will be taken as the average daily temperature. 
                    
    Tmax : float, 
        Maximum daily temperauture (deg C)
            
    Tmax : float, 
        Minimum daily temperauture (deg C)
            
    Tdew : float, optional
        Dew point temperature (deg C). 
                    
    Wspeed : float, reccomended
        Average windspeed in m/s. The default is assumed to be 0.
                     
    Pa : float, optional
        Barometric pressure (kPa), if left as None will be estimated as a 
        function of observation elevation. Nb: should be about 100 kPa. 
                     
    RHmax : float, optional
        Maximum relative humidity (%). The default is 90. 
    
    RHmin : float, optional
        Minimum relative humidity (%). The default is 60. 
                     
    Ea : float, optional
        Actual vapour pressure (kPa). The default is None.
              
    Es : float, optional
        Saturation vapour pressure (kPa). The default is None.

    G : float, optional
        Soil flux density (MJ m^-2 day^-1 ). The default is 0.
              
    Rn : float, optional
        Net radiation at surface (MJ m^-2 day^-1). The default is None.
            
    psy : float, optional
        Psychromatic constant (kPa deg C^-1) . The default is None.

    daylighthours : float, optional
        Measured number of daylight hours. The default is 90 % of total
        possible.

    Raises
    ------
    Exception
        If required parameters are not provided.

    Returns
    -------
    PM: class
        Penman Monteith estimator class 

    """
    def __init__(self, lat, # required, latitude in degrees
                 elev, # required, elevation of site
                 date, # required, day of observations
                 date_format='%Y-%m-%d', # optional, date format if date given as string. 
                 Tmean = None, # optional, average daily temperature in deg C 
                 Tmax = None, # required, max measured temperature in deg C 
                 Tmin = None, # required, min measured temperature in deg C 
                 Tdew = None, # optional, dew point temperature in deg C 
                 Wspeed = None, # reccomended, windspeed in m/s 
                 Pa = None, # optional, atmospheric pressure in kPa 
                 RHmax = None, # reccomended, maximum relative humidity in % 
                 RHmin = None, # reccomended, minimum relative humidity in % 
                 Ea = None, # optional, actual vapour pressure in kPa
                 Es = None, # optional, saturation vapour pressure in kPa
                 G = None, # optional, soil flux density in MJ m^-2 day^-1 ... often set at 0. 
                 Rn = None, # optional, net radiation at surface in MJ m^-2 day^-1
                 psy = None, # optional, Psychromatic constant in kPa deg C^-1 
                 daylighthours = None, # optional, measured number of daylight hours
                 ): 

        
        self.lat = lat * (ma.pi/180)
        self.elev = elev
        # need some estimate of min / max temperature 
        if Tmin is None or Tmax is None: 
            raise Exception('Sorry, need some estimate of min and max temperature')
        self.Tmin = Tmin 
        self.Tmax = Tmax 
        if Tmean is None:
            self.Tmean = (Tmin + Tmax)/2 
        else:
            self.Tmean = Tmean 
        # need some estimate of windspeed really 
        if Wspeed is None: 
            warn('A wind speed estimate is reccomended, setting to 0')
            self.Wspeed = 0 
        else:
            self.Wspeed = Wspeed 
            
        # also need some estimate of relative humidity for estimating Tdew 
        if Tdew is None and RHmax is None:
            warn('A relative humidity estimate is reccomended if dew point is not known')
        if RHmax is None: 
            self.RHmax = 90 
        else:
            self.RHmax = RHmax 
        if RHmin is None:
            self.RHmin = 60
        else:
            self.RHmin = RHmin 
        
        # date handling 
        if isinstance(date,str):
            self.date = datetime.strptime(date, '%Y-%m-%d')
        else:
            self.date = date 
        # get day of year 
        ref_date = datetime(self.date.year-1,12,31)
        td = self.date - ref_date 
        self.doy = td.days 
            
        # set optional properties, which will be computed 
        self.Pa = Pa 
        self.Tdew = Tdew 
        self.Es = Es 
        self.Ea = Ea 
        self.Rn = Rn 
        self.G = G 
        self.psy = psy 
        self.Delta = None # slope vapour pressure curve 
        self.daylight = daylighthours 
        
    
    def getProp(self):
        """
        Estimate / compute all the properties which are required for estimating
        potential evapotranspiration. 

        Returns
        -------
        None.

        """
        if self.Pa is None: 
            self.Pa = self.estPa()
        
        if self.Es is None: 
            self.Es = self.estEs()
        
        if self.Ea is None: 
            self.Ea = self.estEa()
            
        if self.psy is None: 
            self.psy = self.compPsy()
            
        if self.Delta is None: 
            self.Delta = self.compDelta()
            
        if self.G is None: 
            self.G = 0 # close to zero for grass 
            
        if self.Rn is None: 
            self.Rn = self.estRn()
        
            
    def estPa(self):
        b = (293 - (0.0065*self.elev))/293
        return 101.3 * (b**5.26)
    
        
    def estEs(self):
        return (vapPres(self.Tmax)+(vapPres(self.Tmin)))/2 
    
    def estEa(self):
        if self.Tdew is not None: 
            return vapPres(self.Tdew)
        elif isinstance(self.RHmax,float) and isinstance(self.RHmin,float):
            a = vapPres(self.Tmin)*(self.RHmax/100)
            b = vapPres(self.Tmax)*(self.RHmin/100)
            return (a+b)/2 
        else:
            return vapPres(self.Tmin)*(self.RHmax/100)
    
    def compPsy(self):
        return (Cp*self.Pa)/(epsilon*lamda)
    
    def compDelta(self):
        return (4098*vapPres(self.Tmean)) / ((self.Tmean + 273.3)**2)
    
    def estRa(self):
        """
        Estimate extraterrestrial radiation  radiation. 
        This is an involved calculation. 

        Returns
        -------
        Ra: float
            Daily extraterrestrial radiation in MJ m^-2 day^-1
        ws: float 
            Sunset hour angle in radians 

        """
        
        # compute n  days in a year 
        ndaysinyear = 365 
        if self.date.year%4 == 0:
            ndaysinyear = 366 
        # compute inverse of sun / earth distance 
        dr = 1 + (0.0033*ma.cos((ma.pi/ndaysinyear)*self.doy)) 
        # compute solar decimation 
        d = 0.409 * ma.sin(((2*ma.pi*self.doy)/ndaysinyear)-1.39)
        # sunset hour angle 
        ws = ma.acos(ma.atan(self.lat)*ma.tan(d))
        
        # now compute extraterrestrial radiation 
        a = ((24*60)/ma.pi)*Gsolar*dr 
        b = (ws*ma.sin(self.lat)*ma.sin(d)) + (ma.cos(self.lat)*ma.cos(d)*ma.sin(ws))
        Ra = a*b 
        
        return Ra, ws 
    
    def estRsRso(self, n=None, a_s = 0.25, b_s = 0.50):
        """
        Estimate solar radiation. 

        Parameters
        ----------
        n : float, int, optional
            Actual number of daylight hours. The default is 90 % of possible. 
        a_s : float, optional
            Fraction of radiation reaching the earth component. 
            The default is 0.25.
        b_s : float, optional
            Fraction of radiation reaching the earth component. 
            The default is 0.50.

        Returns
        -------
        Rs: float
            solar radiation 
        Rso: float
            clear sky radiation 

        """
        Ra, ws = self.estRa()
        # compute daylight hours 
        N = (24/ma.pi)*ws 
        if n is None: 
            n = 0.6 * N 
        Rs = (a_s + (b_s*(n/N)))*Ra 
        
        Rso = (0.75 + (2e-5 * self.elev))*Ra 
        return Rs, Rso 
    
    def estRnParam(self, n=None):
        """
        Estimate longwave and shortwave radiation 

        Parameters
        ----------
        n : float, int, optional
            Number of measured daylight hours. The default is None.

        Returns
        -------
        Rns : float
            shortwave radiation
        Rnl : TYPE
            longwave radiation 

        """
        Rs, Rso = self.estRsRso(n) # estimate solar radiation and clear sky 
        Rns = (1-alpha)*Rs # shortwave radiation 
        
        avgK = (self.Tmin+273.16+self.Tmax+273.16)/2 # average kelvin temp 
        vapTerm = 0.34 - (0.14*ma.sqrt(self.Ea)) # vapour pressure term 
        radTerm = (1.35*(Rs/Rso)) - 0.35 # radiation term (~ 1)
        
        Rnl = stefboltzman*avgK*vapTerm*radTerm # longwave wave radiation 
        
        return Rns, Rnl 
    
        
    
    def estRn(self):
        """
        Estimate net radiation

        Returns
        -------
        Rn: float 
            Net radiation in MJ m^-2 day^-1 

        """
        # Rn = Rns - Rnl according to Allen et al, so we need to find these 
        # 2 parameters first, shortwave and longwave radiation. 
        
        Rns, Rnl  = self.estRnParam(n=self.daylight)  
        
        return Rns - Rnl 
        
    
    
    def estPET(self):
        """
        Estimate potential evapotranspiration. 

        Returns
        -------
        None.

        """
        self.getProp() # ensure all properties are populated before computing 
        # ET 
        
        # to make the calculation easier to cross reference with Allen et al 
        # i'm breaking it down over multiple lines 
        
        avgK = self.Tmean + 273 # average temp in K 
        na = 0.408 * self.Delta * (self.Rn - self.G) # numerator part a 
        nb = self.psy * (900/avgK) * self.Wspeed*(self.Es - self.Ea)
        
        numon = na + nb # numerator 
        
        denom = self.Delta + (self.psy*(1+(0.34*self.Wspeed))) # denomator 

        Et0 = numon / denom # ET calculation 
        
        if Et0 < 0: 
            return 0 

        return Et0         
        
    
    
class PETarray():
    def __init__(self):
        pass 
    
# test 
# Tmax = 23.00 
# Tmean = 16.38 
# Tmin = 9.61
# Tdew = 13.61
# Wspeed = 8.315 
# Pa = 102.2 
# RHmax = 94 
# date = datetime(2023,8,21)
# lat = 53.8711
# elev = 103 


# pm = PETestimator(lat, elev,
#                   date=date,
#                   Pa=Pa,
#                   # RHmax=RHmax,
#                   Tdew=Tdew, 
#                   Wspeed=Wspeed,
#                   Tmean=Tmean,
#                   Tmax=Tmax,
#                   Tmin=Tmin)

# pm.estPET()





        
    