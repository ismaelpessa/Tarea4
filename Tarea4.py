import numpy as np
import math
import matplotlib.pyplot as plt
import pyfits
import random
import glob
from matplotlib.figure import Figure
import scipy
from scipy import *
from numpy import *
from pylab import *
from numpy.linalg import inv
from scipy import special as sp
from scipy.special import erfinv
from numpy.linalg import cholesky
from scipy.stats import chi2,chisqprob



def random_gen(N,a,b):##Entrega un arreglo de N numeros aleatorios entre a y b
    out=range(N)
    for i in xrange(0,N):
            out[i]=a+(b-a)*random()
    return out

def hist_gen(array,nbins):
    plt.hist(array,nbins)
    #plt.show()

def vector_to_gauss(array,sigma):
    out=range(len(array))
    for i in xrange(0,len(array)):
        out[i]=inv_cum(array[i],sigma)

    return out

def inv_cum(x,sigma):
    return sigma*math.sqrt(2)*erfinv(2.*x-1)

def columnas(filee,x,indicador):
##      Si indicador es 1, columna es float, si es 0, columna es string,si es 2, integer
        n=0
        file=open(filee)
        
        read1=file.readline()
        while len(read1)>1:
            n=n+1
            read1=file.readline()
            
            
        c=range(n)
        file.close()
        file=open(filee)
        
        read=file.readline()
        i=0
        if indicador==1:
            while i<n:
                    
                ID=read.split("\t")
                c[i]=float(ID[x-1])
                i=i+1
                read=file.readline()
        if indicador==0:
            while i<n:
                ID=read.split("\t")
                c[i]=ID[x-1]
                if c[i][len(c[i])-1]=='\n':
                    c[i]=substring(c[i],0,len(c[i])-2)
                i=i+1
                read=file.readline()


        if indicador==2:
            while i<n:
                ID=read.split("\t")
                c[i]=int(float(ID[x-1]))
                i=i+1
                read=file.readline()        
                
        file.close()
        return c
def indexOf(array,n):
        for i in xrange(0,len(array)):
                if array[i]==n:
                        return i
        return -1
def Get_data(data_file):
    t=columnas(data_file,1,1)
    f=columnas(data_file,2,1)
    return t,f


def Matrix(data_file,N):#N = Grado del polinomio+1, la dimension de la matriz es entonces n_datosx(N+1), el numero de parametros sera N+1 considerando al delta
    t1=0.4
    t4=0.7
    H=Get_data(data_file)
    t=H[0]
    f=H[1]
    n=len(f)
    M = [[0 for x in range(N+1)] for y in range(n)] 
    for i in xrange(0,n):
        for j in xrange(0,N+1):
            if j<N:
                M[i][j]=t[i]**j
            if j==N:
                if t[i]<=t4 and t[i]>=t1:#dentro de omega 1
                    M[i][j]=-1
                if t[i]>t4 or t[i]<t1:#fuera de omega 1
                    M[i][j]=0
    return M

def Get_coefs(data_file,N):
    H=Get_data(data_file)
    M=Matrix(data_file,N)
    Mt=np.array(M).transpose()
    M=np.array(M)
    B=Mt.dot(M)
    Binv=inv(B)
    C=Binv.dot(Mt)
    Y=np.array(H[1])
    Sol=C.dot(Y)
    return Sol


def Matrix2(t,f,N):#N = Grado del polinomio+1, la dimension de la matriz es entonces n_datosx(N+1), el numero de parametros sera N+1 considerando al delta
    t1=0.4
    t4=0.7
    
    n=len(f)
    M = [[0 for x in range(N+1)] for y in range(n)] 
    for i in xrange(0,n):
        for j in xrange(0,N+1):
            if j<N:
                M[i][j]=t[i]**j
            if j==N:
                if t[i]<=t4 and t[i]>=t1:#dentro de omega 1
                    M[i][j]=-1
                if t[i]>t4 or t[i]<t1:#fuera de mega 1
                    M[i][j]=0
    return M


def Get_coefs2(t,f,N):
    M=Matrix2(t,f,N)
    Mt=np.array(M).transpose()
    M=np.array(M)
    B=Mt.dot(M)
    Binv=inv(B)
    C=Binv.dot(Mt)
    Y=np.array(f)
    Sol=C.dot(Y)
    return Sol
    
def Plot_Model(data_file,coefs,n): #coefs es el arreglo output de Get_coefs, que tiene los parametros, n es el numero de puntos del modelo
    H=Get_data(data_file)
    t=np.array(H[0])
    f=np.array(H[1])
    N=len(coefs)
    delta=coefs[N-1] #El valor de delta siempre es el ultimo de coefs
    tmod=np.linspace(0.2,0.8,n)
    fmod=range(len(tmod))
    aux=0
    for i in xrange(0,len(tmod)):
        for j in xrange(0,N-1):
            aux=aux+coefs[j]*tmod[i]**j
            if j==N-2:
                z=0
                #print 'Grado = '+str(j)
        
        if tmod[i]>=0.4 and tmod[i]<=0.7:
            fmod[i]=aux-delta
        if tmod[i]<0.4 or tmod[i]>0.7:
            fmod[i]=aux

        aux=0


    plt.plot(t,f,'o',label='data')
    plt.plot(tmod,fmod,label='model')
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.legend()
    plt.savefig('Modelo_data.png')
    plt.show()
    return tmod,fmod,t,f,coefs
            
def squareChi(t,f,coefs,indicador): #Si indicador es 0, no hay transito, si indicador es 1, si hay transito
    N=len(coefs)
    delta=indicador*coefs[N-1] #El valor de delta siempre es el ultimo de coefs
    square_chi=0
    aux=0
    var=np.var(f)
    n=len(f)
    fmod=range(n)
    for i in xrange(0,n):
            for j in xrange(0,N-1):
                    aux=aux+coefs[j]*t[i]**j
            if t[i]>=0.4 and t[i]<=0.7:
                    fmod[i]=aux-delta
            if t[i]<0.4 or t[i]>0.7:
                    fmod[i]=aux
            aux=0

    for i in xrange(0,n):
            square_chi=square_chi+(f[i]-fmod[i])**2

    z=range(len(f))
    for i in xrange(0,len(z)):
        z[i]=f[i]-fmod[i]
    z=np.array(z)
    sigma=(3.E-5)**2
    M = [[0 for x in range(len(f))] for y in range(len(f))] 
    for i in xrange(0,len(f)):
        for j in xrange(0,len(f)):
            if j==i:
                M[i][j]=sigma




    #plt.plot(t,fmod)
    #plt.plot(t,f)
    #plt.show()
    #print (z.transpose().dot(inv(M))).dot(z)
    return (z.transpose().dot(inv(M))).dot(z)
         
    
    
   
    #print var
    var=(3.E-5)**2
    return square_chi/var
            
            
def Generate_data_no_transit(coefs,n):#Simula n datos con los coeficientes dados, asumiendo delta=0, y agregando errores gaussianos
        t_out=np.linspace(0.2,0.8,n)
        f_out=range(n)
        aux=0
        delta=0
        N=len(coefs)
        for i in xrange(0,n):
                for j in xrange(0,N-1):
                        aux=aux+coefs[j]*t_out[i]**j
                if t_out[i]>=0.4 and t_out[i]<=0.7:
                        f_out[i]=aux-delta
                if t_out[i]<0.4 or t_out[i]>0.7:
                        f_out[i]=aux
                aux=0
        rando=random_gen(n,0,1)
        errors=vector_to_gauss(rando,3.E-5)
        fout_err=range(n)
        for i in xrange(0,n):
                fout_err[i]=f_out[i]+errors[i]
        #plt.plot(t_out,f_out)
        #plt.plot(t_out,fout_err,'o')
        #plt.show()
        return t_out,fout_err

def Generate_data_transit(coefs,n):#Simula n datos con los coeficientes dados, asumiendo delta=0, y agregando errores gaussianos
        t_out=np.linspace(0.2,0.8,n)
        f_out=range(n)
        aux=0
        N=len(coefs)
        delta=coefs[N-1]
        for i in xrange(0,n):
                for j in xrange(0,N-1):
                        aux=aux+coefs[j]*t_out[i]**j
                if t_out[i]>=0.4 and t_out[i]<=0.7:
                        f_out[i]=aux-delta
                if t_out[i]<0.4 or t_out[i]>0.7:
                        f_out[i]=aux
                aux=0
        rando=random_gen(n,0,1)
        errors=vector_to_gauss(rando,3.E-5)
        fout_err=range(n)
        for i in xrange(0,n):
                fout_err[i]=f_out[i]+errors[i]
        #plt.plot(t_out,f_out)
        #plt.plot(t_out,fout_err,'o')
        #plt.show()
        return t_out,fout_err

def Get_pvalue(x,y,squarechi):#x,y corresponden a la distribucin chi cuadrado, squarechi al punto desde el que se debe integrar
        p_value=0
        deltax=x[1]-x[0]
        I=0
        for i in xrange(0,len(x)):
                if x[i]>=squarechi:
                        I=I+deltax*y[i]
        return I
                
def Get_pvalue2(chisq,df):
    return chisqprob(chisq,df)
        
def Generate_squareChi(df):
        x=np.linspace(chi2.ppf(0.0,df),chi2.ppf(0.99999,df),100000)
        y=chi2.pdf(x,df)
        return x,y
def Generate_data_multi_no_transit(coefs_mod,n,N,coefs_dat):#N data sets, each with n data points
        df=n-len(coefs_mod)
        CHI=Generate_squareChi(df)
        x=CHI[0]
        y=CHI[1]
        squareChis=range(N)
        pvalues=range(N)
        for i in xrange(0,N):
                print i
                DAT=Generate_data_no_transit(coefs_dat,n)
                t=DAT[0]
                f=DAT[1]
                squareChis[i]=squareChi(t,f,coefs_mod,0)
                pvalues[i]=Get_pvalue(x,y,squareChis[i])
                pvalues[i]=Get_pvalue2(squareChis[i],df)
        return pvalues,squareChis

def Generate_data_multi_transit(coefs_mod,n,N,coefs_dat):#N data sets, each with n data points
        
        df=n-len(coefs_mod)
        CHI=Generate_squareChi(df)
        x=CHI[0]
        y=CHI[1]
        squareChis=range(N)
        pvalues=range(N)
        for i in xrange(0,N):
                DAT=Generate_data_transit(coefs_dat,n)
                t=DAT[0]
                f=DAT[1]
                squareChis[i]=squareChi(t,f,coefs_mod,1)
                pvalues[i]=Get_pvalue(x,y,squareChis[i])
        return pvalues,squareChis
                


def split_array(t,f,K,j):##Separa los arreglos t y f en K sub arreglos, aproximadamente homogeneos, y entega el j-esimo (del 1 al n)
        n=range(int((len(t)/K)+1))
        cont=0
        for i in xrange(0,len(t)):
                for l in xrange(0,len(n)):
                        if i==j+n[l]*K-1:
                                cont=cont+1
        out_t=range(cont)
        out_f=range(cont)
        cont=0
        for i in xrange(0,len(t)):
                for l in xrange(0,len(n)):
                        if i==j+n[l]*K-1:
                                out_t[cont]=t[i]
                                out_f[cont]=f[i]
                                cont=cont+1
        return out_t,out_f
        
                        


def extract_sub_array(subt,subf,t,f):##Extrae los arreglos subt y subf de t y f respectivamente y entrega 2 arreglos de dimensiones len(t)-len(subt)
        subn=len(subt)
        n=len(t)
        n_out=n-subn
        out_f=range(n_out)
        out_t=range(n_out)
        j=0
        for i in xrange(0,n):
                if indexOf(subt,t[i])==-1:
                        out_t[j]=t[i]
                        out_f[j]=f[i]
                        j=j+1
        return out_t,out_f
                
        
def Cross_validation(t,f,N,K): #N es el numero de terminos del polinomio, N-1 es el grado y N+1 el numero total de parametros,t y f de los datos, K es el kfold de la cross validation
        Err=range(K)
        for j in xrange(1,K+1):
                H=split_array(t,f,K,j)
                t_test=H[0]
                f_test=H[1]
                coefs=Get_coefs2(t_test,f_test,N)
                B=extract_sub_array(t_test,f_test,t,f)
                t_train=B[0]
                f_train=B[1]
                Err[j-1]= squareChi(t_train,f_train,coefs,1)
                #Plot_Model('datos.dat',coefs,10000)
                

        
        return sum(Err)
def Cross_validation_Test(t,f,K,Grad_mayor):
        Cross_vec=range(Grado_mayor)
        Param_n=range(Grado_mayor)
        for i in xrange(1,Grado_mayor+1):  
                print i
                Cross_vec[i-1]= Cross_validation(t,f,i,K)
                Param_n[i-1]=i-1 ##Param_n es el grado del polinomio
        
        
        
        plt.plot(Param_n,Cross_vec,'o',label='Cross validation')
        plt.legend()
        plt.title('Cross_Validation Test')
        plt.xlabel('Grado del polinomio')
        plt.savefig('Cross_valid.png')
        plt.show()




def Likelihood(t,f,coefs):
        L=-squareChi(t,f,coefs,1)/2.  ##Log like=L
        return L
        
def AIC(t,f,coefs):
        N=len(t)
        return -2.*Likelihood(t,f,coefs)+2*len(coefs)+2*len(coefs)*(len(coefs)+1)/(N-len(coefs)-1)
def Test_AIC(Grado_mayor,data):
        H=Get_data(data)
        t=np.array(H[0])
        f=np.array(H[1])
        AIC_vec=range(Grado_mayor)
        Param_n=range(Grado_mayor)
        for i in xrange(1,Grado_mayor+1):  
                print i
                A=Get_coefs(data,i) ##i corresponde al numero de terminos del polinomio, considerando el termino de grado cero, es decir, i-1=grado del polinomio, y, contando el delta, i+1 corresponde al numero de parametros
                #Plot_Model(data,A,10000)
                AIC_vec[i-1]=AIC(t,f,A)
                Param_n[i-1]=i-1
        plt.plot(Param_n,AIC_vec,'o',label='AIC')
        plt.legend()
        plt.title('AIC_test')
        plt.xlabel('Grado del polinomio')
        plt.savefig('AIC.png')
        plt.show()
        minAIC=min(AIC_vec)
        index=indexOf(AIC_vec,minAIC)
        minP=Param_n[index]
        print 'El minimo AIC es para '+str(minP)
        print 'Esto significa que el grado optimo del plinomio sera '+str(minP)
                
                
        return
                
def BIC(t,f,coefs):
        N=len(t)
        return -2.*Likelihood(t,f,coefs)+2*len(coefs)+len(coefs)*np.log(N)
def Test_BIC(Grado_mayor,data):
        H=Get_data(data)
        t=np.array(H[0])
        f=np.array(H[1])
        BIC_vec=range(Grado_mayor)
        Param_n=range(Grado_mayor)
        for i in xrange(1,Grado_mayor+1):
                print i
                A=Get_coefs(data,i) ##i corresponde al numero de terminos del polinomio, considerando el termino de grado cero, es decir, i-1=grado del polinomio, y, contando el delta, i+1 corresponde al numero de parametros
                #Plot_Model(data,A,10000)
                BIC_vec[i-1]=AIC(t,f,A)
                Param_n[i-1]=i-1
        plt.plot(Param_n,BIC_vec,'o',label='BIC')
        plt.title('BIC_Test')
        plt.legend()
        plt.xlabel('Grado del polinomio')
        plt.savefig('BIC.png')
        plt.show()
        minBIC=min(BIC_vec)
        index=indexOf(BIC_vec,minBIC)
        minP=Param_n[index]
        print 'El minimo BIC es para '+str(minP)
        print 'Esto significa que el grado optimo del polinomio sera '+str(minP)
                
                
        
def P3(data,Grado_mayor,K):
     Test_AIC(Grado_mayor,data)
     Test_BIC(Grado_mayor,data)
     H=Get_data(data)
     t=np.array(H[0])
     f=np.array(H[1])
     Cross_validation_Test(t,f,K,Grado_mayor)
     
     
     
    
        
def P1(data):
        A=Get_coefs(data,6)
        Plot_Model(data,A,10000)
        H=Get_data(data)
        sqrchi=squareChi(np.array(H[0]),np.array(H[1]),A,1)
        n=len(H[0])
        N=len(A)
        df=n-N
        CHI=Generate_squareChi(df)
        x=CHI[0]
        y=CHI[1]
        pval=Get_pvalue(x,y,sqrchi)
        print 'p-value= '+str(pval)
        print 'SquareChi_P1='+str(sqrchi)
        plt.close()
        plt.title('Square_Chi Distribution, degrees of freedom: '+str(df))
        plt.plot(x,y)
        plt.axvline(x=sqrchi, ymin=0., ymax = 0.7, linewidth=2, color='k')
        plt.xlabel('Square_Chi')
        plt.ylabel('Probability density function')
        texto1='p-value='+str(pval)
        texto2='square-chi='+str(sqrchi)
        plt.text(100,0.004,texto1)
        plt.text(100,0.002,texto2)
        plt.savefig('Square_Chi_dist.png')
        plt.show()
        return
        
def P2(data):
        N=1000
        n=300
        nbin=70              
               
        
        A=Get_coefs(data,6)
        Dat=Generate_data_multi_no_transit(A,n,N,A)
        pvalues=Dat[0]
        CHIS=Dat[1]
        plt.title('Modelo: Polinomio grado 5 sin transito, p-value promedio='+str(mean(pvalues)))
        hist_gen(pvalues,nbin)
        plt.savefig('Poli5_notransit.png')
        #plt.show()
        plt.close()

        B=Get_coefs(data,4)
        Dat=Generate_data_multi_no_transit(B,n,N,A)
        pvalues=Dat[0]
        CHIS=Dat[1]
        plt.title('Modelo: Polinomio grado 3 sin transito, p-value promedio='+str(mean(pvalues)))
        hist_gen(pvalues,nbin)
        plt.savefig('Poli3_notransit.png')
        plt.close()

        B=Get_coefs(data,5)
        Dat=Generate_data_multi_no_transit(B,n,N,A)
        pvalues=Dat[0]
        CHIS=Dat[1]
        plt.title('Modelo: Polinomio grado 4 sin transito, p-value promedio='+str(mean(pvalues)))
        hist_gen(pvalues,nbin)
        plt.savefig('Poli3_notransit.png')
        plt.close()

        
        C=Get_coefs(data,1)
        Dat=Generate_data_multi_no_transit(C,n,N,A)
        pvalues=Dat[0]
        CHIS=Dat[1]
        plt.title('Modelo: Polinomio grado 0 sin transito, p-value promedio='+str(mean(pvalues)))
        hist_gen(pvalues,nbin)
        plt.savefig('Poli0_notransit.png')
        plt.close()

        D=Get_coefs(data,8)
        Dat=Generate_data_multi_no_transit(D,n,N,A)
        pvalues=Dat[0]
        CHIS=Dat[1]
        plt.title('Modelo: Polinomio grado 7 sin transito, p-value promedio='+str(mean(pvalues)))
        hist_gen(pvalues,nbin)
        plt.savefig('Poli7_notransit.png')
        plt.close()

        D=Get_coefs(data,9)
        Dat=Generate_data_multi_no_transit(D,n,N,A)
        pvalues=Dat[0]
        CHIS=Dat[1]
        plt.title('Modelo: Polinomio grado 8 sin transito, p-value promedio='+str(mean(pvalues)))
        hist_gen(pvalues,nbin)
        plt.savefig('Poli8_notransit.png')
        plt.close()

        At=Get_coefs(data,6)
        Dat=Generate_data_multi_transit(At,n,N,At)
        pvalues=Dat[0]
        CHIS=Dat[1]
        plt.title('Modelo: Polinomio grado 5 con transito, p-value promedio='+str(mean(pvalues)) )
        hist_gen(pvalues,nbin)
        plt.savefig('Poli5_transit.png')
        plt.close()

        return



        
        
        
        
        
        
        
        
        
        
        
        
        
   
        
         
         


    
    
        
  
data='datos.dat'
K=5
Grado_mayor=8

P3(data,Grado_mayor,K)

P1(data)
P2(data)

