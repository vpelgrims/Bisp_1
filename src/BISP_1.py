# -*- coding: utf-8 -*-
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   BISP-1
#
#   Bayesian Inference of Starlight Polarization in 1D
#
# --- 
#
#   Implementation of the method to perform the decomposition of optical
#   polarization of stars with known distances along distance as presented in
#   Pelgrims et al. 2022, A&A, ..., ...
#
#   @author: V.Pelgrims
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from astropy.io import ascii
from astropy.table import Table

from dynesty import utils as dyfunc
import dynesty
import corner

import bisp1_logll as logL

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Stars(object):
    '''
        Object Stars.
        Loads and formats a star data sample and allow for quick visualization.
    '''
    def __init__(self,starSample):
        '''
            Initialization of the object from input.
            Input must be either
            - a (7,nStar) array with (q,u,plx,s_q,s_u,c_qu,s_plx)
            - or a filename pointing to a catalog in csv or ascii format.
            ! units !
                plx and s_plx must be in [mas]
                polarization q,u and uncertainties in fraction. (1 = 100%)
        '''
        if isinstance(starSample,str):
            starSample = Stars._getArray_from_Cat(starSample)
        #
        self.stararray = starSample
        self.plx_min = np.min(starSample[2])
        self.plx_max = np.max(starSample[2])
        self.qu_max = np.max([np.abs(starSample[0]),np.abs(starSample[1])])
        self.nStar = len(starSample[0])
        self.NoLayerLL = Stars._get_NoLayerLL(starSample)
        #
    #

    def _getArray_from_Cat(catname):
        '''
            Opens a catalog of starlight polarization data and returns
            an array for later uses. The names of the columns need to follow
            'q_obs','u_obs','plx','s_q','s_u','s_plx' (and 'c_qu', option)
        '''
        data_ = ascii.read(catname)
        #
        # check all the required fields are present in the data file
        required = ['q_obs','u_obs','s_q','s_u','plx','s_plx']
        if all(req in data_.keys() for req in required):
            pass
        else:
            message = ("Missing keys or wrong spelling in data table. \n "+\
                        "There should be: 'q_obs','u_obs','s_q','s_u','plx','s_plx' " +\
                            "with, optionally, 'c_qu'.")
            raise ValueError(message)
        #
        qV = np.asarray(data_['q_obs'])
        uV = np.asarray(data_['u_obs'])
        e_qV = np.asarray(data_['s_q'])
        e_uV = np.asarray(data_['s_u'])
        plx_obs = np.asarray(data_['plx'])       # plx in mas
        e_plx = np.asarray(data_['s_plx'])       # s_plx in mas
        #
        the_array = np.asarray([qV,uV,plx_obs,e_qV,e_uV,0*e_qV,e_plx])
        if 'c_qu' in data_.keys():
            the_array[5] = np.asarray(data_['c_qu'])
        #
        return the_array
    
    def _get_NoLayerLL(starSample):
        '''
            Compute the (log-)likelihood of the data given no dust layer,
            i.e. assuming that the measured polarization are due to noise
            only.
        '''
        stararray = starSample
        qs = stararray[0]; us = stararray[1]
        Cqq = stararray[3]**2 ; Cuu = stararray[4]**2 ; Cqu = stararray[5]
        detC = Cqq * Cuu - Cqu**2
        invCqq = Cuu / detC
        invCuu = Cqq / detC
        invCqu = - Cqu /detC
        chi2 = qs**2 * invCqq + us**2 * invCuu + 2 * qs * us * invCqu
        
        ll_i = np.sum(np.log(np.exp(-.5*chi2)/(2*np.pi*np.sqrt(detC))))
        return ll_i
    


    #@classmethod
    def showData_QUMU(self,**kwargs):
        '''
            Generates a (q,u),mu plot of the data.
            **kwargs={ylim,xlim,show_error,usetex}
        '''
        qs,us,plxs = self.stararray[:3,:]
        plxs = plxs/1000.   # convert plx from [mas] to [as]
        dms = 5.*np.log10(1./plxs) - 5. # compute distance modulus
        qs = 100. * np.copy(qs) ; us = 100. * np.copy(us)  # convert pol to [%]
        # preparing a figure frame with ad hoc size and ranges.
        plt.figure()
        plt.plot(dms,qs,'+',markersize=2,color='C0')
        plt.plot(dms,us,'+',markersize=2,color='C1')
        plt.gca().set_ylim(ymin = np.minimum(np.percentile(qs,1),\
                                    np.percentile(us,1)) - 0.2,\
                            ymax = np.maximum(np.percentile(qs,99),\
                                np.percentile(us,99)) + 0.2)
        ylims = plt.gca().get_ylim()
        xlims = plt.gca().get_xlim()
        #
        usetex = True
        show_error = True
        if kwargs is not None:
            for key,val in kwargs.items():
                if key == 'ylim':
                    ylims = val
                elif key == 'xlim':
                    xlims = val
                elif key == 'show_error':
                    show_error = val
                elif key == 'usetex':
                    usetex = val
                else:
                    raise ('Invalid arguments. Ignored.')

        plt.clf()
        #
        if show_error == False:
            plt.plot(dms,qs,'o',markerfacecolor='none',markersize=4,\
                        markeredgewidth=1.5,color='darkgreen',alpha=.5,\
                            label=r'$q_{\rm{V}}$ obs.')
            plt.plot(dms,us,'d',markerfacecolor='none',markersize=4,\
                        markeredgewidth=1.5,color='blue',alpha=.5,\
                            label=r'$u_{\rm{V}}$ obs.')
        elif show_error == True:
            #
            dms_bl = 5*(np.log10(1000./(self.stararray[2]+self.stararray[6]))-1)
            dms_up = 5*(np.log10(1000./(self.stararray[2]-self.stararray[6]))-1)
            qs = self.stararray[0]*100
            us = self.stararray[1]*100
            s_qs = self.stararray[3]*100
            s_us = self.stararray[4]*100
            plt.errorbar(x=dms,y=qs,xerr=[dms-dms_bl,dms_up-dms],\
                            yerr=s_qs,fmt='o',markerfacecolor='none',\
                            markersize=4,markeredgewidth=1.5,alpha=.25,\
                            color='darkgreen',capsize=0,elinewidth=1,\
                            label=r'$q_{\rm{V}}$ obs.');
            plt.errorbar(x=dms,y=us,xerr=[dms-dms_bl,dms_up-dms],\
                            yerr=s_us,fmt='d',markerfacecolor='none',\
                            markersize=4,markeredgewidth=1.5,alpha=.25,\
                            color='blue',capsize=0,elinewidth=1,\
                            label=r'$u_{\rm{V}}$ obs.');
            plt.plot(dms,qs,'o',markerfacecolor='none',markersize=4,\
                            markeredgewidth=1.5,color='darkgreen',alpha=.5)
            plt.plot(dms,us,'d',markerfacecolor='none',markersize=4,\
                            markeredgewidth=1.5,color='blue',alpha=.5)
            #

        # print the cosmetic
        plt.gca().set_ylim(ylims)
        plt.gca().set_xlim(xlims)
        plt.xlabel(r'$\mu$ Distance Modulus',fontsize=16,usetex=usetex)
        plt.ylabel(r'$(q_{\rm{V}},u_{\rm{V}})$ Stellar Stokes [\%]',fontsize=16,\
                    usetex=usetex)
        #
        return
    #

    def showData_QUscatter(self,**kwargs):
        '''
            Generates a scatter plot of the (q,u) data including uncertainties
            and coloured according to distance (modulus)
            **kwargs={cmin,cmax,usetex}
        '''

        mu = 5*(np.log10(1000./self.stararray[2])-1.)
        qs,us,s_qs,s_us = self.stararray[[0,1,3,4],:] * 100.

        plt.figure();
        ax = plt.gca();

        cmin = np.min(mu) ; cmax = np.max(mu)
        usetex = True
        if kwargs is not None:
            for key,val in kwargs.items():
                if key == 'cmin':
                    cmin = val
                elif key == 'cmax':
                    cmax = val
                elif key == 'usetex':
                    usetex = val
                else:
                    raise ('Invalid arguments. Ignored.')
        #
        s_c = np.copy(mu)
        s_c[mu<=cmin] = cmin
        s_c[mu>=cmax] = cmax
        s = ax.scatter(qs,us,c=s_c);

        norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
        d_color = np.array([(mapper.to_rgba(v)) for v in s_c])

        for q,u,eq,eu,color in zip(qs,us,s_qs,s_us,d_color):
            plt.plot(q, u, 'o', color=color)
            plt.errorbar(q, u, xerr=eq,yerr=eu, lw=1, capsize=3, color=color)

        cb = plt.colorbar(s)
        cb.set_label(r'$\mu$ Distance Modulus')
        plt.axis('square')
        plt.xlabel(r'$q_{\rm{V}}$ [\%] Stellar Stokes',fontsize=16,usetex=True)
        plt.ylabel(r'$u_{\rm{V}}$ [\%] Stellar Stokes',fontsize=16,usetex=True)
        plt.show()
        #
        return
    #

#

class LayerPriors:
    '''
        Object LayerPriors.
        Used to determine types and values of the priors for each of the six
        parameters characterizing one layer.
        
        Parameters are: plx, q, u, Cqq, Cuu, Cqu.
        
        _type must be 'Flat' or 'Gauss' for uniform or Gaussian priors.
        
        _val1 and _val2 are minimum and maximum if prior type = 'Flat'
        _val1 and _val2 are mean and stddev is prior type = 'Gauss'
        
        **Note** that for the plx parameters, values in the input array are
        reverted for the 'Flat' prior (historical reason)
    '''
    def __init__(self,types,values):
        self.plx_type = types[0]
        if types[0] == 'Flat':
            self.plx_val2 = values[0]
            self.plx_val1 = values[6]
        elif types[0] == 'Gauss':
            self.plx_val1 = values[0]
            self.plx_val2 = values[6]
        else:
            print('outch')
        #
        self.q_type = types[1]
        self.q_val1 = values[1]
        self.q_val2 = values[7]
        #
        self.u_type = types[2]
        self.u_val1 = values[2]
        self.u_val2 = values[8]
        #
        self.Cqq_type = types[3]
        self.Cqq_val1 = values[3]
        self.Cqq_val2 = values[9]
        #
        self.Cuu_type = types[4]
        self.Cuu_val1 = values[4]
        self.Cuu_val2 = values[10]
        #
        self.Cqu_type = types[5]
        self.Cqu_val1 = values[5]
        self.Cqu_val2 = values[11]
        #
#
    
class ModelPriors:
    '''
        Object ModelPriors.
        Gather LayerPriors object to build models with given name
        corresponding to a given number of layers along sightlines.
        
        Functions getTypeList() and getValueList() return the list of list
        of list with types and values for each parameters for each layer
        for a given model.
    '''
    def __init__(self,modelName,types,values):
        if modelName == 'OneLayer':
         self.layer_1 = LayerPriors(types[0],values[0])
         self.nlayers = 1
        elif modelName == 'TwoLayers':
            self.layer_1 = LayerPriors(types[0],values[0])
            self.layer_2 = LayerPriors(types[1],values[1])
            self.nlayers = 2
        elif modelName == 'ThreeLayers':
            self.layer_1 = LayerPriors(types[0],values[0])
            self.layer_2 = LayerPriors(types[1],values[1])
            self.layer_3 = LayerPriors(types[2],values[2])
            self.nlayers = 3
        elif modelName == 'FourLayers':
            self.layer_1 = LayerPriors(types[0],values[0])
            self.layer_2 = LayerPriors(types[1],values[1])
            self.layer_3 = LayerPriors(types[2],values[2])
            self.layer_4 = LayerPriors(types[3],values[3])
            self.nlayers = 4
        elif modelName == 'FiveLayers':
            self.layer_1 = LayerPriors(types[0],values[0])
            self.layer_2 = LayerPriors(types[1],values[1])
            self.layer_3 = LayerPriors(types[2],values[2])
            self.layer_4 = LayerPriors(types[3],values[3])
            self.layer_5 = LayerPriors(types[4],values[4])
            self.nlayers = 5
        else:
            raise ValueError('unknown modelName')
        #

    #
    def getTypeList(self):
        ty_list = []
        def getTyp(tylist,theL):
            tylist.append([theL.plx_type,theL.q_type,theL.u_type,\
                            theL.Cqq_type,theL.Cuu_type,theL.Cqu_type])
            return tylist

        if self.nlayers>=1:
            ty_list = getTyp(ty_list,self.layer_1)
        if self.nlayers>=2:
            ty_list = getTyp(ty_list,self.layer_1)
        if self.nlayers>=3:
            ty_list = getTyp(ty_list,self.layer_1)
        if self.nlayers>=4:
            ty_list = getTyp(ty_list,self.layer_1)
        if self.nlayers>=5:
            ty_list = getTyp(ty_list,self.layer_1)
        #
        return ty_list


    def getValueList(self):
        va_list = []
        def getVal(valist,theL):
            if theL.plx_type == 'Flat':
                valist.append([theL.plx_val2,theL.q_val1,theL.u_val1,\
                                theL.Cqq_val1,theL.Cuu_val1,theL.Cqu_val1,\
                                theL.plx_val1,theL.q_val2,theL.u_val2,\
                                    theL.Cqq_val2,theL.Cuu_val2,theL.Cqu_val2])
            else: # plx_type = 'Gauss'
                valist.append([theL.plx_val1,theL.q_val1,theL.u_val1,\
                                theL.Cqq_val1,theL.Cuu_val1,theL.Cqu_val1,\
                                theL.plx_val2,theL.q_val2,theL.u_val2,\
                                    theL.Cqq_val2,theL.Cuu_val2,theL.Cqu_val2])
            return valist
        #
        if self.nlayers>=1:
            va_list = getVal(va_list,self.layer_1)
        if self.nlayers>=2:
            va_list = getVal(va_list,self.layer_2)
        if self.nlayers>=3:
            va_list = getVal(va_list,self.layer_3)
        if self.nlayers>=4:
            va_list = getVal(va_list,self.layer_4)
        if self.nlayers>=5:
            va_list = getVal(va_list,self.layer_5)
        #
        return va_list


class Priors:
    def __init__(self,stars,inputTypes=None,inputValues=None):
        '''
            Initialization of the object.
            Default (using `stars`) or using the types and values given in
            optional arguments.
            Input:
                - `stars`: a Stars object
                - inputTypes: a list (up to 5 elements) of lists (number of
                                layer) of list (length of 6 for all 6 model
                                parameters per layer) to specify the type of
                                priors.
                - inputValues: a list (up to 5 elements) of lists (number of
                                layer) of list (length of 12 for all 6 model
                                parameters per layer) to specify the range of
                                priors or mean and stddev.
                    see updatePriors()

            **Note  **: the default flat prior on cloud parallaxes are defined
                        by the stars. By default the decomposition will search
                        for clouds within the range encompased by the stars'
                        plx. This might not be the case in general!
            **Note 2**: There is an hard-coded threshold of minimum 5 stars
                        between each cloud, when more than one cloud are to be
                        considered. Be careful if you intend to change this and
                        make sure this is also propagated to bisp_logll.
        '''

        plx_s = np.sort(stars.stararray[2])

        def feedit(nclouds):
            typ = []
            val = []
            for nn in range(nclouds):
                typ.append(['Flat','Flat','Flat','Flat','Flat','Flat'])
                val.append([stars.plx_max,-stars.qu_max,-stars.qu_max,\
                                    0.,0.,-1.e-4,\
                                    stars.plx_min,stars.qu_max,stars.qu_max,\
                                        1.e-4,1.e-4,1.e-4])
                val[nn][6] = plx_s[5*(nclouds-1-nn)]
                # the '5' is our minimum allowed stars in between clouds.
                # It is necessary to comply with computation of prior_transform
                # in bisp_logll.
            return typ,val

        # OneLayer:
        ty,va = feedit(1)
        self.OneLayer = ModelPriors('OneLayer',ty,va)

        # TwoLayers:
        ty,va = feedit(2)
        self.TwoLayers = ModelPriors('TwoLayers',ty,va)

        # ThreeLayers:
        ty,va = feedit(3)
        self.ThreeLayers = ModelPriors('ThreeLayers',ty,va)

        # FourLayers:
        ty,va = feedit(4)
        self.FourLayers = ModelPriors('FourLayers',ty,va)

        # FiveLayers:
        ty,va = feedit(5)
        self.FiveLayers = ModelPriors('FiveLayers',ty,va)

        # update priors if needed
        self.updatePriors(inputTypes=inputTypes,inputValues=inputValues)


    #
    def updatePriors(self,inputTypes=None,inputValues=None):
        '''
            Update the default priors according to the inputs.
        
            Inputs are list of list (of list):
                [list_for_oneLayerModel,list_for_twoLayerModel,...]
            with
                list_for_oneLayerModel = [list_of_param]
                list_for_twoLayerModel = [list_for_1st_layer,list_for_2nd_layer]
                ...

            Parameters are read in the order: plx,q,u,Cqq,Cuu,Cqu.
            Values must be specified as:
            [max_plx,min_q,min_u,min_Cqq,min_Cuu,min_Cqu,
                min_plx,max_q,max_u,max_Cqq,max_Cuu,max_Cqu] if all 'Flat'
            or
            [mean_plx,mean_q,mean_u,mean_Cqq,mean_Cuu,mean_Cqu,
                std_plx,std_q,std_u,std_Cqq,std_Cuu,std_Cqu] if all 'Gauss'.

            **Note**: the inverted order of min and max for the plx in case of
            'Flat'.
            
        '''
        if inputTypes is not None:
            for mm in range(len(inputTypes)):
                n = len(inputTypes[mm])
                if n == 1:
                    values = self.OneLayer.getValueList()
                    self.OneLayer = ModelPriors('OneLayer',inputTypes[mm],\
                                                    values)
                elif n == 2:
                    values = self.TwoLayers.getValueList()
                    self.TwoLayers = ModelPriors('TwoLayers',inputTypes[mm],\
                                                    values)
                elif n == 3:
                    values = self.ThreeLayers.getValueList()
                    self.ThreeLayers = ModelPriors('ThreeLayers',\
                                                    inputTypes[mm],values)
                elif n == 4:
                    values = self.FourLayers.getValueList()
                    self.FourLayers = ModelPriors('FourLayers',inputTypes[mm],\
                                                    values)
                elif n == 5:
                    values = self.FiveLayers.getValueList()
                    self.FiveLayers = ModelPriors('FiveLayers',inputTypes[mm],\
                                                    values)
                else:
                    raise ('Invalid size of priorType inputs')
        if inputValues is not None:
            for mm in range(len(inputValues)):
                if n == 1:
                    types = self.OneLayer.getTypeList()
                    self.OneLayer = ModelPriors('OneLayer',types,\
                                                        inputValues[mm])
                elif n == 2:
                    types = self.TwoLayers.getTypeList()
                    self.TwoLayers = ModelPriors('TwoLayers',types,\
                                                        inputValues[mm])
                elif n == 3:
                    types = self.ThreeLayers.getTypeList()
                    self.ThreeLayers = ModelPriors('ThreeLayers',types,\
                                                        inputValues[mm])
                elif n == 4:
                    types = self.FourLayers.getTypeList()
                    self.FourLayers = ModelPriors('FourLayers',types,\
                                                        inputValues[mm])
                elif n == 5:
                    types = self.FiveLayers.getTypeList()
                    self.FiveLayers = ModelPriors('FiveLayers',types,\
                                                        inputValues[mm])
                else:
                    raise ('Invalid size of priorType inputs')
        return
    #

    def printPriors(self,modelName,numformat='%.4f'):
        '''
            Looks into the prior types and values of a given model and prints
            in the terminal a table with parameter names, types of priors and
            values used to specify the functional forms for each parameter and
            each layer.
            - modelName: str to specify the name of the model for which priors
                            are of interest
            - numformat: (optional), str to specific the numerical format to
                            display numerical values.
        '''
        if modelName == 'OneLayer':
            Nclouds = int(1)
            tab_type = np.asarray(self.OneLayer.getTypeList())
            tab_vals = np.asarray(self.OneLayer.getValueList())
            #
        elif modelName == 'TwoLayers':
            Nclouds = int(2)
            tab_type = np.asarray(self.TwoLayers.getTypeList())
            tab_vals = np.asarray(self.TwoLayers.getValueList())
            #
        elif modelName == 'ThreeLayers':
            Nclouds = int(3)
            tab_type = np.asarray(self.ThreeLayers.getTypeList())
            tab_vals = np.asarray(self.ThreeLayers.getValueList())
            #
        elif modelName == 'FourLayers':
            Nclouds = int(4)
            tab_type = np.asarray(self.FourLayers.getTypeList())
            tab_vals = np.asarray(self.FourLayers.getValueList())
            #
        elif modelName == 'FiveLayers':
            Nclouds = int(5)
            tab_type = np.asarray(self.FiveLayers.getTypeList())
            tab_vals = np.asarray(self.FiveLayers.getValueList())
            #
        else:
            message = ("Wrong model name. It should be one of: \n",\
                        "'OneLayer', 'TwoLayers', 'ThreeLayers', ",\
                            "'FourLayers', 'FiveLayers'.")
            raise ValueError(message)
        #
        rows = []
        [rows.append('Cloud '+str(nn)) for nn in range(Nclouds)]
        
        table_type = Table([rows,tab_type[:,0],tab_type[:,1],tab_type[:,2],\
                                tab_type[:,3],tab_type[:,4],tab_type[:,5]],\
                            names=['Clouds','plx','q','u','Cqq','Cuu','Cqu'],\
                            dtype=['U8','U8','U8','U8','U8','U8','U8'])



        table_vals = Table([rows,tab_vals[:,0],tab_vals[:,1],tab_vals[:,2],\
                                tab_vals[:,3],tab_vals[:,4],tab_vals[:,5],\
                                tab_vals[:,6],tab_vals[:,7],tab_vals[:,8],\
                                tab_vals[:,9],tab_vals[:,10],tab_vals[:,11]],\
                            names=['Clouds','plx_v1','q_v1','u_v1',\
                                    'Cqq_v1','Cuu_v1','Cqu_v1',\
                                    'plx_v2','q_v2','u_v2',\
                                    'Cqq_v2','Cuu_v2','Cqu_v2'],\
                            dtype=['U8','f8','f8','f8','f8','f8','f8',\
                                        'f8','f8','f8','f8','f8','f8'])
        #
        print('\n Prior Types: \n ------------ \n')
        print(table_type,'\n')
        
        for key in table_vals.keys()[1:]:
            table_vals[key].format = numformat
        print('\n Prior Values: \n ------------- \n')
        print(table_vals,'\n')
        print("For 'Flat' priors 'v1' and 'v2' refer to minimum and maximum",\
                "\n except for 'plx' where there are inverted as compared to",\
                    "plx_val1 and plx_val2.")
        print("For 'Gauss' priors 'v1' and 'v2' refer to mean and stddev. \n")
#


class Decomp:
    '''
    Object Decomp.
    Wraps the dynesty sampler and allows for specific access and manipulations
    '''
    def __init__(self,dySampler):
        '''
            Initialization is simply the wrap.
        '''
        self.sampler = dySampler
    #

    def PosteriorSamples(self,samplerSize=10000):
        '''
            Reads dyresults and return weighted posterior samples
            Optional argument: 'samplerSize' specifies the maximum number of
                        posterior samples to return.
        '''
        dyresults = self.sampler.results
        samples = dyresults.samples
        weights = np.exp(dyresults.logwt-dyresults.logl[-1])
        weights /= np.sum(weights)
        posteriors = dyfunc.resample_equal(samples,weights)
        return posteriors[:samplerSize,:]
    #
    def SummaryStat(self):
        '''
            Reads dyresults and returns evidence, error on evidence,
            and maximum log-likelihood
        '''
        dyresults = self.sampler.results
        Z_evidence = dyresults.logz[-1]
        err_Z_evidence = dyresults.logzerr[-1]
        max_logl = dyresults.logl[-1]
        return Z_evidence,err_Z_evidence,max_logl
    
    def Values_at_MaxLikelihood(self):
        '''
            Extracts parameter values at maximum log-likelihood
        '''
        dyresults = self.sampler.results
        bf_values = dyresults.samples[-1]
        return bf_values
    
    def CloudProperties(self,samplesSize=10000):
        '''
            Extracts some cloud properties from the posterior distributions.
            Posteriors are treated as if they were Gaussian. Only the mean
            and standard deviation are reported.
            Outputs an astropy table.
        '''
        
        samples = self.PosteriorSamples(samplesSize)
        Nclouds = int(samples.shape[1]/6)
        
        tab = []
        for nn in range(Nclouds):
            m_plx = np.mean(samples[:,0+nn*6])
            s_plx = np.std(samples[:,0+nn*6])
            m_q = np.mean(samples[:,1+nn*6])
            s_q = np.std(samples[:,1+nn*6])
            m_u = np.mean(samples[:,2+nn*6])
            s_u = np.std(samples[:,2+nn*6])

            Cov = np.cov(samples[:,1+nn*6],samples[:,2+nn*6])
            if np.linalg.det(Cov) <= 0.0:
                # if it does not work, add a little noise!
                Cov *= (1+ 1.e-4*np.random.randn(4).reshape(2,2))
            
            s = np.asarray([m_q,m_u])
            dMaha_0 = np.sqrt(np.dot(s,np.dot(np.linalg.inv(Cov),s)))
            tab_nn = [m_plx,m_q,m_u,s_plx,s_q,s_u,dMaha_0]
            tab.append(tab_nn)
        tab = np.asarray(tab)

        rows = []
        [rows.append('Cloud '+str(nn)) for nn in range(Nclouds)]
        table = Table([rows,tab[:,0],tab[:,3],tab[:,1],\
                        tab[:,4],tab[:,2],tab[:,5],tab[:,6]],\
                        names=['Clouds','plxC','err_plxC','qC','err_qC',\
                                'uC','err_uC','dMaha0'],\
                        dtype=['U8','f8','f8','f8','f8','f8','f8','f8'])
        return table
    
    def printCloudProperties(self,samplesSize=10000,numformat='%.4f'):
        '''
            Print in the terminal the astropy table outputed by CloudProperties
        '''
        propTable = self.CloudProperties(samplesSize=samplesSize)
        #
        propTable['plxC'].format = numformat
        propTable['err_plxC'].format = numformat
        propTable['qC'].format = numformat
        propTable['err_qC'].format = numformat
        propTable['uC'].format = numformat
        propTable['err_uC'].format = numformat
        propTable['dMaha0'].format = numformat
        print(propTable)
        return

    def showPosteriors(self,samplesSize=10000):
        '''
            Prints on the front a corner plot of all the model parameters.
        '''
        samples = self.PosteriorSamples(samplesSize)
        ndim = samples.shape[1]
        Nclouds = int(ndim/6)
        labels = [r'$\varpi_{\mathrm{C}}$ [mas]',\
                    r'$q_{\mathrm{C}}$ [$\%$]',\
                    r'$u_{\mathrm{C}}$ [$\%$]',\
                    r'$C_{qq}^{\mathrm{int}}$ [$\%^2$]',\
                    r'$C_{uu}^{\mathrm{int}}$ [$\%^2$]',\
                    r'$C_{qu}^{\mathrm{int}}$ [$\%^2$]']
        plt.rcParams['font.size'] = 9
        #
        f_size = np.minimum(Nclouds*6,9.5)
        f = plt.figure(figsize=(f_size,f_size))
        # convert polarization in [%]
        for ci in range(Nclouds):
            samples[:,[1+6*ci,2+6*ci]] *= 100
            samples[:,[3+6*ci,4+6*ci,5+6*ci]] *= 10000
        #
        corner.corner(samples,smooth=1.,color='C0',\
                        labels=labels*Nclouds,\
                        label_kwargs={'fontsize':6,'usetex':True},\
                        quantiles=[0.16,0.5,0.86],\
                        show_titles=True,\
                        title_kwargs={'fontsize':6,'usetex':True},\
                        fig=f,labelpad=.04,plot_datapoints=False)
        plt.show()
        return
    
    def showPosteriors_meanPol(self,samplesSize=10000):
        '''
            Same as showPosteriors but only show (plx,q,u) for each layer.
            The parameters characterizing the intrinsic scatter are dropped.
        '''
        samples = self.PosteriorSamples(samplesSize)
        ndim = samples.shape[1]
        Nclouds = int(ndim/6)
        labels = [r'$\varpi_{\mathrm{C}}$ [mas]',\
                    r'$q_{\mathrm{C}}$ [$\%$]',\
                    r'$u_{\mathrm{C}}$ [$\%$]']
        plt.rcParams['font.size'] = 12
        #
        f_size = np.minimum(Nclouds*6,9.5)
        f = plt.figure(figsize=(f_size,f_size))
        # convert polarization in [%]
        # and extract required columns
        cols = []
        for ci in range(Nclouds):
            samples[:,[1+6*ci,2+6*ci]] *= 100
            cols = np.concatenate((cols,ci*6+np.arange(3)))
        samples = samples[:,cols.astype('int')]
        #
        corner.corner(samples,smooth=1.,color='C0',\
                        labels=labels*Nclouds,\
                        label_kwargs={'fontsize':8,'usetex':True},\
                        quantiles=[0.16,0.5,0.86],\
                        show_titles=True,\
                        title_kwargs={'fontsize':8,'usetex':True},\
                        fig=f,labelpad=.04,plot_datapoints=False)
        plt.show()
        return

    def overPlot_QUMUClouds(self,**kwargs):
        '''
            Overplots on an existing figure the results for each layer
            in the (q,u)-mu plane with 2.5,16,50,84,97.5 percentiles.
        '''
        #
        samples = self.PosteriorSamples()
        Nclouds = int(samples.shape[1]/6)
        
        cols = np.arange(Nclouds) * 6
        
        plx_c = samples[:,cols]/1000.   # cloud's plx in as
        
        q_c = np.cumsum(samples[:,1+cols]*100.,axis=1)  # cumulative of clouds' q Stokes in %
        u_c = np.cumsum(samples[:,2+cols]*100.,axis=1)  # cumulative of clouds' u Stokes in %

        xlims = plt.gca().get_xlim()
        ylims = plt.gca().get_ylim()
        
        # look at the distributions
        plx_ = np.percentile(plx_c,[97.5,84,50,16,2.5],axis=0).T
        dmsC_ = 5.*np.log10(1./plx_)-5. # distance modulus of clouds
        
        qC_ = np.percentile(q_c,[2.5,16,50,84,97.5],axis=0).T
        uC_ = np.percentile(u_c,[2.5,16,50,84,97.5],axis=0).T
        
        # prepare for plotting
        if Nclouds == 1:
            x = np.asarray([dmsC_[0,4],xlims[1]])[None,:]
        else:
            x = np.asarray([dmsC_[:,4],np.concatenate((dmsC_[1:,0],np.asarray([xlims[1]])))]).T
        #
        if np.median(dmsC_[:,2])<xlims[0]:
            x[0,0] = xlims[0]
            dmsC_[0,:-1] *= 0
        else:
            pass
        #
        for nn in range(Nclouds):
            # print q and u
            plt.fill_between(x[nn],qC_[nn,0],qC_[nn,-1],
                                color='C2',alpha=.25,linewidth=.5,linestyle=':')
            plt.fill_between(x[nn],qC_[nn,1],qC_[nn,-2],
                                color='C2',alpha=.25,linewidth=.5,linestyle='--')
            #
            plt.fill_between(x[nn],uC_[nn,0],uC_[nn,-1],
                                color='C4',alpha=.25,linewidth=.5,linestyle=':')
            plt.fill_between(x[nn],uC_[nn,1],uC_[nn,-2],
                                color='C4',alpha=.25,linewidth=.5,linestyle='--')
            # print mu_c
            plt.fill_betweenx(ylims,dmsC_[nn,0],dmsC_[nn,-1],
                                color='C3',alpha=.2,linewidth=1.5,linestyle=':')
            plt.fill_betweenx(ylims,dmsC_[nn,1],dmsC_[nn,-2],
                                color='C3',alpha=.2,linewidth=1.5,linestyle='--')
            #
            if nn == 0:
                plt.plot(x[nn],[qC_[nn,2],qC_[nn,2]],
                                color='C2',linewidth=1,label=r'$q_{\rm{C}}$ mod.')
                plt.plot(x[nn],[uC_[nn,2],uC_[nn,2]],
                                color='C4',linewidth=1,label=r'$u_{\rm{C}}$ mod.')
                #
                plt.plot([dmsC_[nn,2],dmsC_[nn,2]],ylims,
                                color='C3',linewidth=1,label=r'$\mu_{\rm{C}}$ mod.')
                #
            else:
                plt.plot(x[nn],[qC_[nn,2],qC_[nn,2]],
                                color='C2',linewidth=1)
                plt.plot(x[nn],[uC_[nn,2],uC_[nn,2]],
                                color='C4',linewidth=1)
                #
                plt.plot([dmsC_[nn,2],dmsC_[nn,2]],ylims,
                                color='C3',linewidth=1)
                #
            #
        #
        return
    #


class Bisp:
    '''
        Object Bisp
        Main object. Contains information on star data and priors and allows
        for the application of the maximum-likelihood analysis of the data
        for the different models (assumed different number of layers along the
        LOS) and to access the results and compare the performance of the
        different models.
    '''
    def __init__(self,stars,priors,**kwargs):
        '''
            Initialization from input:
            - stars: a Stars object
            - priors: a Priors object
            - **kwargs: used to control ```dynesty``` settings
        '''
        self.stars = stars
        self.priors = priors
        self.testedModel = []
        #
        # settings for dynesty inference  - - - - - - - - - - - - - -
        self.default_dynestycontrol()
        if kwargs is not None:
            self.update_dynestycontrol(kwargs)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
    # Call and run the different possible models:
    # One-layer model:
    # ---------------
    def run_OneLayer(self,fromFresh=True):
        '''
            Run the 1-layer model on the stars using the priors
        '''
        # setup the sampling experiment
        # Point to the right log-likelihood function
        def loglike(theta):
            modelname = '1Layer'
            return logL.def_model(theta,modelname,self.stars.stararray)
        # Point to the right prior function
        def prior_transform(unibox):
            return logL.def_prior_IntSc(unibox,\
                                        self.priors.OneLayer.getTypeList(),\
                                        self.priors.OneLayer.getValueList(),\
                                        self.stars.stararray)
        # initialization of dynesty sampler
        init_dyn = self.dynesty_settings
        # check if existing previous run has been performed
        if ((not hasattr(self,'OneLayer')) or fromFresh):
            # if not, initialize it
            sampler = dynesty.NestedSampler(loglike,prior_transform,6,\
                                                nlive=init_dyn['nlive'],\
                                                sample='auto')
        else:
            # if yes, load it to start from there
            sampler = self.OneLayer.sampler
        #
        # run dynesty sampler
        sampler.run_nested(dlogz=init_dyn['dlogz'],\
                            maxiter=init_dyn['maxiter'],\
                                print_progress=not(init_dyn['silent']))
        # wrappe the result as a `Decomp` object
        Result = Decomp(sampler)
        # add the results
        self.OneLayer = Result
        #
        if 'OneLayer' not in self.testedModel:
            self.testedModel += ['OneLayer']
        #
        return

    # Two-layer model:
    # ----------------
    def run_TwoLayers(self,fromFresh=True):
        '''
            Run the 2-layers model on the stars using the priors
        '''
        # setup the sampling experiment
        # Point to the right log-likelihood function
        def loglike(theta):
            modelname = '2Layers'
            return logL.def_model(theta,modelname,self.stars.stararray)
        # Point to the right prior function
        def prior_transform(unibox):
            return logL.def_prior_IntSc(unibox,\
                                        self.priors.TwoLayers.getTypeList(),\
                                        self.priors.TwoLayers.getValueList(),\
                                        self.stars.stararray)
        # initialization of dynesty sampler
        init_dyn = self.dynesty_settings
        # check if a run has been performed previously
        if ((not hasattr(self,'TwoLayers')) or fromFresh):
            # if not, initialize it
            sampler = dynesty.NestedSampler(loglike,prior_transform,2*6,\
                                                nlive=init_dyn['nlive'],\
                                                sample='auto')
        else:
            # if yes, load it to start from there
            sampler = self.TwoLayers.sampler
        #
        # run dynesty sampler
        sampler.run_nested(dlogz=init_dyn['dlogz'],\
                            maxiter=init_dyn['maxiter'],\
                                print_progress=not(init_dyn['silent']))
        # wrappe the result as a `Decomp` object
        Result = Decomp(sampler)
        # add the results
        self.TwoLayers = Result
        #
        if 'TwoLayers' not in self.testedModel:
            self.testedModel += ['TwoLayers']
        #
        return

    # Three-layer model:
    # ------------------
    def run_ThreeLayers(self,fromFresh=True):
        '''
            Run the 3-layers model on the stars using the priors
        '''
        # setup the sampling experiment
        # Point to the right log-likelihood function
        def loglike(theta):
            modelname = '3Layers'
            return logL.def_model(theta,modelname,self.stars.stararray)
        # Point to the right prior function
        def prior_transform(unibox):
            return logL.def_prior_IntSc(unibox,\
                                        self.priors.ThreeLayers.getTypeList(),\
                                        self.priors.ThreeLayers.getValueList(),\
                                        self.stars.stararray)
        # initialization of dynesty sampler
        init_dyn = self.dynesty_settings
        # check if a run has been performed previously
        if ((not hasattr(self,'ThreeLayers')) or fromFresh):
            # if not, initialize it
            sampler = dynesty.NestedSampler(loglike,prior_transform,3*6,\
                                                nlive=init_dyn['nlive'],\
                                                sample='auto')
        else:
            # if yes, load it to start from there
            sampler = self.ThreeLayers.sampler
        #
        # run dynesty sampler
        sampler.run_nested(dlogz=init_dyn['dlogz'],\
                            maxiter=init_dyn['maxiter'],\
                                print_progress=not(init_dyn['silent']))
        # wrappe the result as a `Decomp` object
        Result = Decomp(sampler)
        # add the results
        self.ThreeLayers = Result
        #
        if 'ThreeLayers' not in self.testedModel:
            self.testedModel += ['ThreeLayers']
        #
        return

    # Four-layer model:
    # ------------------
    def run_FourLayers(self,fromFresh=True):
        '''
            Run the 4-layers model on the stars using the priors
        '''
        # setup the sampling experiment
        # Point to the right log-likelihood function
        def loglike(theta):
            modelname = '4Layers'
            return logL.def_model(theta,modelname,self.stars.stararray)
        # Point to the right prior function
        def prior_transform(unibox):
            return logL.def_prior_IntSc(unibox,\
                                        self.priors.FourLayers.getTypeList(),\
                                        self.priors.FourLayers.getValueList(),\
                                        self.stars.stararray)
        # initialization of dynesty sampler
        init_dyn = self.dynesty_settings
        # check if a run has been performed previously
        if ((not hasattr(self,'FourLayers')) or fromFresh):
            # if not, initialize it
            sampler = dynesty.NestedSampler(loglike,prior_transform,4*6,\
                                                nlive=init_dyn['nlive'],\
                                                sample='auto')
        else:
            # if yes, load it to start from there
            sampler = self.FourLayers.sampler
        #
        # run dynesty sampler
        sampler.run_nested(dlogz=init_dyn['dlogz'],\
                            maxiter=init_dyn['maxiter'],\
                                print_progress=not(init_dyn['silent']))
        # wrappe the result as a `Decomp` object
        Result = Decomp(sampler)
        # add the results
        self.FourLayers = Result
        #
        if 'FourLayers' not in self.testedModel:
            self.testedModel += ['FourLayers']
        #
        return

    
    # Five-layer model:
    # ------------------
    def run_FiveLayers(self,fromFresh=True):
        '''
            Run the 5-layers model on the stars using the priors
        '''
        # setup the sampling experiment
        # Point to the right log-likelihood function
        def loglike(theta):
            modelname = '5Layers'
            return logL.def_model(theta,modelname,self.stars.stararray)
        # Point to the right prior function
        def prior_transform(unibox):
            return logL.def_prior_IntSc(unibox,\
                                        self.priors.FiveLayers.getTypeList(),\
                                        self.priors.FiveLayers.getValueList(),\
                                        self.stars.stararray)
        # initialization of dynesty sampler
        init_dyn = self.dynesty_settings
        # check if a run has been performed previously
        if ((not hasattr(self,'FiveLayers')) or fromFresh):
            # if not, initialize it
            sampler = dynesty.NestedSampler(loglike,prior_transform,5*6,\
                                                nlive=init_dyn['nlive'],\
                                                sample='auto')
        else:
            # if yes, load it to start from there
            sampler = self.FiveLayers.sampler
        #
        # run dynesty sampler
        sampler.run_nested(dlogz=init_dyn['dlogz'],\
                            maxiter=init_dyn['maxiter'],\
                                print_progress=not(init_dyn['silent']))
        # wrappe the result as a `Decomp` object
        Result = Decomp(sampler)
        # add the results
        self.FiveLayers = Result
        #
        if 'FiveLayers' not in self.testedModel:
            self.testedModel += ['FiveLayers']
        #
        return

    # # - - - - # #

    def GetSummaryStat(self):
        '''
            Searches for already evaluated models and extract the summary
            statistics: evidence, error on evidence, AIC, and also compute
            the probability of each tested model that it is actually the
            best model (in the pool of available model) that minimizes the
            loss of information.
            Returns an astropy table.
        '''
        testedModel = []
        sumStats = []
        
        # by default, consider NoLayer model
        testedModel.append('NoLayer')
        stats = [np.nan,np.nan,self.stars.NoLayerLL]
        stats = np.concatenate((stats,[2*0 - 2* stats[2]]))
        sumStats.append(stats)
        #
        #
        if hasattr(self,'OneLayer'):
            testedModel.append('OneLayer')
            stats = np.asarray(self.OneLayer.SummaryStat())
            stats = np.concatenate((stats,[2*6 - 2*stats[2]]))
            sumStats.append(stats)
        if hasattr(self,'TwoLayers'):
            testedModel.append('TwoLayers')
            stats = np.asarray(self.TwoLayers.SummaryStat())
            stats = np.concatenate((stats,[2*12 - 2*stats[2]]))
            sumStats.append(stats)
        if hasattr(self,'ThreeLayers'):
            testedModel.append('ThreeLayers')
            stats = np.asarray(self.ThreeLayers.SummaryStat())
            stats = np.concatenate((stats,[2*18 - 2*stats[2]]))
            sumStats.append(stats)
        if hasattr(self,'FourLayers'):
            testedModel.append('FourLayers')
            stats = np.asarray(self.FourLayers.SummaryStat())
            stats = np.concatenate((stats,[2*24 - 2*stats[2]]))
            sumStats.append(stats)
        if hasattr(self,'FiveLayers'):
            testedModel.append('FiveLayers')
            stats = np.asarray(self.FiveLayers.SummaryStat())
            stats = np.concatenate((stats,[2*30 - 2*stats[2]]))
            sumStats.append(stats)
        #
        if sumStats != []:
            sumStats = np.asarray(sumStats)
            # sumStats[:,3] records the AIC obtained for each model
            # from whose probabilities to minimizing the information loss are get:
            P_jm = np.exp((np.min(sumStats[:,3]) - sumStats[:,3])/2.)
    
            table = Table([testedModel,sumStats[:,0],sumStats[:,1],sumStats[:,2],\
                            sumStats[:,3],P_jm],\
                            names=['Model','Z','errZ','max-LL','AIC','Pbest'],\
                            dtype=['U16','f8','f8','f8','f8','f8'])
        else:
            table = Table([['None'],[0],[0],[0],[0],[0]],\
                            names=['Model','Z','errZ','max-LL','AIC','Pbest'],\
                            dtype=['U16','f8','f8','f8','f8','f8'])
        #
        return table

    def printSummaryStat(self,numformat='%.4f'):
        '''
            Prints on the terminal the outputed table from GetSummaryStat().
            numformat (optional) is used to specify the numerical format to
            be used while printing.
        '''
        sumTable = self.GetSummaryStat()
        sumTable['Z'].format = numformat
        sumTable['errZ'].format = numformat
        sumTable['max-LL'].format = numformat
        sumTable['AIC'].format = numformat
        sumTable['Pbest'].format = numformat
        print(sumTable)
        return
    
    def writeSummaryStat(self,filename,numformat='%.4f'):
        '''
            Writes on the disk the output of GetSUmmaryStat()
            Input:
                - filename: str to specify the location and name of the file
                            where to write the file.
                - numformat: to specify the numerical format to be used
        '''
        sumTable = self.GetSummaryStat()
        sumTable['Z'].format = numformat
        sumTable['errZ'].format = numformat
        sumTable['max-LL'].format = numformat
        sumTable['AIC'].format = numformat
        sumTable['Pbest'].format = numformat
        if filename[-4:] != '.csv':
            filename + '.csv'
        ascii.write(sumTable,comment='#',delimiter=',',\
                        overwrite=True,output = filename)
        return
    
    def update_dynestycontrol(self,settingList):
        '''
            Updates the list of settings to control the behavior of dynesty.
            Input:
                - settingList: a dictionary to update specific variables
        '''
        prev_settings = self.dynesty_settings
        for key,val in settingList.items():
            if key in prev_settings:
                prev_settings[key] = val
            #
        #
        self.dynesty_settings = prev_settings
        return

    def default_dynestycontrol(self):
        '''
            Sets the default settings to control the nested sampling
            experiments using dynesty.
        '''
        self.dynesty_settings = {'nlive':500,'maxiter':10000,'dlogz':0.1,\
                                    'silent':True}
        return


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# That's all!                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
