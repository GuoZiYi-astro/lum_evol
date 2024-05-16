'''
This is the python code try to calculate the evolution of the luminosity
weighted abundance evolution using empirical function of stellar luminosity.
In this code, we do not include any chemical evolution. All the evolution of
metallicity and elemental abundance are from other real galactic chemical
evolution model. By inputting the star formation history, the stellar initial
mass function, the stellar lifetime, the evolution of metallicity, the
evolution of the elemental abundance, and the function of luminosity, you can
finally get the evolution of the total luminosity, and the evolution of the
luminosity weighted abundance, which is always observed from the stacked
stellar spectrum.

In this version, luminosity do not evolve with the stellar evolution. We only
use the main-sequence luminosity. 

Ziyi Guo
15, May, 2024
'''

import numpy as np
import os
import time as t_module
from scipy.interpolate import interp1d
from scipy import integrate
from matplotlib import pyplot as plt

class lum_abu(object):
    '''
    Parameters:
    ------------------

    dt:  float
    
    This is the time step of the evolution. The unit is year.

    Default: 3e6


    dm:  float

    This is the mass step of the stellar initial mass function. The unit is
    Msun.

    Default: 0.01


    tend:  float

    This is the total evolution time. The unit is year.

    Default: 13e9


    sfh:  2d-array

    This is the list of the star formation history. sfh[:][0] should be the
    time point, sfh[:][1] should be the star formation rate at the
    corresponding time. In the code will do the linearly interpolation.

    Default: [[0,1], [13e9,1]]


    imf_type:  string

    This is to select the type of the stellar initial mass function. There are
    'kroupa' (Kroupa 2001),''. After this definition, you can still use the parameter
    'alpha', 'imf_bdys', 'mass_lim' to modify the specific IMF.

    Default: 'kroupa'


    alpha:  3 element list

    This is the alpha for kroupa IMF. The list should contain 3 elements, for
    the three part of the IMF.

    Default: [-0.3, -1.3, -2.3]


    imf_bdys:  2 element list

    This is the boundary of the IMF. 

    Default: [0.08, 100]


    mass_lim:  2 element list

    This is the transition mass in the IMF for kroupa IMF.

    Default: [0.08, 0.5]


    lum_func:  string

    This is the name of empirical function. Please choose one to use:
    'duric_slaris' (Duric 2004 for low-mass star, Slaris 2006 for stars more
    massive than 2 Msun, this function do not contain the metallicity), ''. 

    Default: 'duric_slaris'


    lifetime_file:  string

    This is the position and name of the lifetime table file. The newly formed
    lifetime table should follow the rule in the 'lifes' file, and then can be
    read by the code.

    Default: '/lifes/life_schaller_92.txt'


    z_evol:  2D list

    This is the evolution of the metallicity. Using this, we can calculate the
    luminosity with the impact of metallicity. The list type is the same with
    the sfh.

    Default: [[0,0], [13e9, 0.02]]

    '''

    def __init__(self, dt = 3e6, dm = 0.01, tend = 13e9, \
            sfh = [[0,1], [13e9, 1]], \
            imf_type = 'kroupa', alpha = [-0.3, -1.3, -2.3], \
            imf_bdys = [0.08, 100], mass_lim = [0.08, 0.5], \
            lum_func = 'duric_slaris', \
            lifetime_file = '/lifes/life_schaller_92.txt', \
            z_evol = [[0,0], [13e9, 0.02]]):
        # transfer the default parameters to self.
        self.dt = dt
        self.dm = dm
        self.tend = tend
        self.sfh = np.array(sfh)
        self.imf_type = imf_type
        self.alpha = alpha
        self.imf_bdys = imf_bdys
        self.mass_lim = mass_lim
        self.lum_func = lum_func
        self.lifetime_file = lifetime_file
        self.z_evol = np.array(z_evol)

        print('Stars start to shinning')
        start_time = t_module.time()

        # get the path of this file
        self.path = os.getcwd()
        self.lifetime_file = self.path+self.lifetime_file

        # get the time, sfr, mass
        self.times_start = np.arange(0, tend, dt)
        self.times_mid = self.times_start+0.5*dt
        self.nt = len(self.times_start)
        self.sfr = np.interp(self.times_mid, self.sfh[:,0], self.sfh[:,1])
        self.star_mass = np.arange(self.imf_bdys[0], self.imf_bdys[1], self.dm)
        self.metallicity = np.interp(self.times_mid, self.z_evol[:,0], self.z_evol[:,1])

        # self.A is the normalization parameter of imf number
        if self.imf_type == 'kroupa':
            self.A = self.norm_kroupa_imf()
            imf_list = self.__get_kroupa_n_list()
            self.imf_list = self.A*imf_list

        # read the lifetimes and get the final list
        self.__read_and_interp_lifetime_file()

        # calculate the luminosity list
        self.__cal_lum_for_step()

        # transfer the lifetimes from year to step
        self.__cal_life_step()

        # initialize the luminosity evolution track list
        self.lum_step = np.zeros((self.nt, self.nt))
        for it in range(self.nt):
            self.__timestep(it)
        end_time = t_module.time()

        # calculate the total luminosity
        self.__cal_total_lum()
        print('   Shinning end. Using ' + str(round((end_time-start_time), 2)) + ' s.')
        self.plot_lum_evolution()

    def __timestep(self, it):
        # Calculate the number of stars of every stellar mass bin in this step
        A_it = self.sfr[it]*self.dt
        star_it = A_it*self.imf_list
        lum_list_it = self.lums_list[:,it]
        lum_step = np.zeros(self.nt)
        for im, mass in enumerate(self.star_mass):
            n_life, r_final = divmod(self.life_step[im, it],1)
            n_life = int(n_life)
            if n_life == 0:
                lum_step[it] += star_it[im]*r_final*lum_list_it[im]
            elif it+n_life >= self.nt:
                lum_step[it:] += star_it[im]*lum_list_it[im]
            else:
                in_life = n_life+it
                lum_step[it:in_life] += star_it[im]*lum_list_it[im]
                lum_step[in_life] += star_it[im]*lum_list_it[im]*r_final
        self.lum_step[:,it] = lum_step

    def __cal_total_lum(self):
        total_lum = np.zeros(self.nt)
        for it in range(self.nt):
            total_lum[it] = sum(self.lum_step[it,:])
        self.total_lum = total_lum

    def __cal_lum_for_step(self):
        lum_func = self.lum_func
        star_mass = self.star_mass
        dm = self.dm
        Z = self.metallicity
        lums = np.zeros((len(star_mass), self.nt))
        if lum_func == 'duric_slaris':
            for im,m in enumerate(star_mass):
                lums[im,0] = integrate.quad(self.duric_slaris, m, m+dm, limit = 50)[0]
            for iz in range(self.nt-1):
                lums[:,iz+1] = lums[:,0]
        self.lums_list = lums

    def duric_slaris(self, m):
        if m < 0.421:
            lum = 0.23*np.power(m, 2.3)
        elif m < 1.96:
            lum = np.power(m,4)
        elif m < 55.41:
            lum = 1.4*np.power(m,3.5)
        else:
            lum = 32000*m
        return lum
    
    def plot_main_sequence_lums(self, Z):
        # This function will plot the luminosity of the MS stars formed in the
        # nearest metallicity to Z 
        z_list = self.metallicity
        abs_z = []
        for iz,z in enumerate(z_list):
            abs_z.append(abs(Z-z))
        idx = abs_z.index(min(abs_z))
        lum_list_it = self.lums_list[:,idx]
        star_mass = self.star_mass
        plt.plot(star_mass, lum_list_it)
        plt.xlabel(r'Stellar mass [M$_\odot$]')
        plt.ylabel(r'Luminosity [L$_\odot$]')
        plt.title('Z = '+str(Z))

    def plot_IMF_weighted_main_sequence(self, it):
        lum_list_it = self.lums_list[:,it]
        star_it = self.imf_list
        lums = star_it*lum_list_it
        star_mass = self.star_mass
        Z = self.metallicity[it]
        plt.plot(star_mass, lums)
        plt.xlabel(r'Stellar mass [M$_\odot$]')
        plt.ylabel(r'Luminosity [L$_\odot$]')
        plt.title('Z = '+str(Z))

    def __cal_life_step(self):
        lifetimes = self.lifetimes_list
        dt = self.dt
        life_step = lifetimes/dt
        self.life_step = life_step

    def __read_and_interp_lifetime_file(self):
        lifetime_file = self.lifetime_file
        ori_lifetimes = np.loadtxt(lifetime_file)
        lifetimes = ori_lifetimes[1:,1:]
        m_list = ori_lifetimes[1:,0]
        z_list = ori_lifetimes[0,1:] 
        mid_m_lifetime = np.zeros((len(self.star_mass), len(z_list)))
        for iz in range(len(z_list)):
            mid_m_lifetime[:,iz] = 10**np.interp(self.star_mass, m_list, np.log10(lifetimes[:,iz]))
        final_lifetime_list = np.zeros((len(self.star_mass), self.nt))
        for im in range(len(self.star_mass)):
            final_lifetime_list[im,:] = 10**np.interp(self.metallicity, z_list, np.log10(mid_m_lifetime[im,:]))
        self.lifetimes_list = final_lifetime_list

    # Calculate the number list for every stellar mass bin
    def __get_kroupa_n_list(self):
        stars = self.star_mass
        dm = self.dm
        imf_list = np.zeros(len(stars))
        for im in range(len(stars)):
            imf_list[im] = integrate.quad(self.kroupa_imf, stars[im], stars[im]+dm, limit = 50)[0]
        return imf_list
        
    def norm_kroupa_imf(self):
        # normalize the IMF to 1 solar mass
        imf_m = integrate.quad(self.kroupa_imf_m, self.imf_bdys[0], self.imf_bdys[1], limit = 50)
        return 1/imf_m[0]

    def kroupa_imf_m(self, m):
        '''
        This is the definition of the kroupa imf mass function. This function has not been normalized to number one.

        Input parameteres:
        m : the mass of stars, actually is the low limit of the mass bin.

        Return: 
        n : the number of this mass in this mass bin.
        '''
        alpha1 = self.alpha[0]
        alpha2 = self.alpha[1]
        alpha3 = self.alpha[2]
        m1 = self.mass_lim[0]
        m2 = self.mass_lim[1]
        ml = self.imf_bdys[0]
        mh = self.imf_bdys[1]

        a1 = m1**(alpha1-alpha2)
        a2 = a1*m2**(alpha3-alpha2)
        if m >= ml and m < m1:
            return m*m**alpha1
        elif m >= m1 and m < m2:
            return m*a1*m**alpha2
        elif m >= m2 and m < mh:
            return m*a2*m**alpha3
        else: 
            print('Out of boundary')
            return 0


    def kroupa_imf(self, m):
        '''
        This is the definition of the kroupa imf number function. This function has not been normalized to number one.

        Input parameteres:
        m : the mass of stars, actually is the low limit of the mass bin.

        Return: 
        n : the number of this mass in this mass bin.
        '''
        alpha1 = self.alpha[0]
        alpha2 = self.alpha[1]
        alpha3 = self.alpha[2]
        m1 = self.mass_lim[0]
        m2 = self.mass_lim[1]
        ml = self.imf_bdys[0]
        mh = self.imf_bdys[1]

        a1 = m1**(alpha1-alpha2)
        a2 = a1*m2**(alpha3-alpha2)
        if m >= ml and m < m1:
            return m**alpha1
        elif m >= m1 and m < m2:
            return a1*m**alpha2
        elif m >= m2 and m < mh:
            return a2*m**alpha3
        else: 
            print('Out of boundary')
            return 0

    def plot_lum_evolution(self, **kwargs):
        x_time = self.times_start
        y_lum = self.total_lum
        plt.plot(x_time, y_lum, **kwargs)
        plt.xlabel('Evolution time [yr]')
        plt.ylabel(r'Luminosity [L$_\odot$]')
        plt.title('Total luminosity evolution track')

    def cal_lum_average_abu(self, ism_ele = [[0,0],[13e9, 0.1]]):
        lums = self.lum_step
        lum_sum = self.total_lum
        ele_lums = np.zeros((self.nt, self.nt))
        ism_ele = np.array(ism_ele)
        ele_evol = np.interp(self.times_mid, ism_ele[:,0], ism_ele[:,1])
        for it in range(self.nt):
            ele_lums[:,it] = lums[:,it]*ele_evol[it]
        ele_sum = np.zeros(self.nt)
        for it in range(self.nt):
            ele_sum[it] = sum(ele_lums[it,:])
        ele_abu = ele_sum/lum_sum
        return ele_abu

    def plot_lum_average_abu(self, ism_ele = [[0,0], [13e9, 0.1]], **kwargs):
        abu = self.cal_lum_average_abu(ism_ele=ism_ele)
        plt.plot(self.times_start, abu, **kwargs)
        plt.xlabel('Evolution time [yr]')
        plt.ylabel('Luminosity averaged abundance')


