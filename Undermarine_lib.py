import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import io as skio


def viewimage(im,normalise=True,MINI=0.0, MAXI=255.0):
    """ Cette fonction fait afficher l'image EN NIVEAUX DE GRIS 
        dans gimp. Si un gimp est deja ouvert il est utilise.
        Par defaut normalise=True. Et dans ce cas l'image est normalisee 
        entre 0 et 255 avant d'être sauvegardee.
        Si normalise=False MINI et MAXI seront mis a 0 et 255 dans l'image resultat
        
    """
    imt=np.float32(im.copy())
    if platform.system()=='Darwin': 
        prephrase='open -a GIMP '
        endphrase=' ' 
    elif platform.system()=='Windows':  
        prephrase='"C:/Program Files/GIMP 2/bin/gimp-2.10.exe" '
        endphrase=' '
    else: 
        prephrase='gimp '
        endphrase= ' &'
    
    if normalise:
        m=im.min()
        imt=imt-m
        M=imt.max()
        if M>0:
            imt=255*imt/M

    else:
        imt=(imt-MINI)/(MAXI-MINI)
        imt[imt<0]=0
        imt[imt>1]=1
        imt *= 255
    
    nomfichier=tempfile.mktemp('TPIMA.png')
    commande=prephrase +nomfichier+endphrase
    imt = imt.astype(np.uint8)
    skio.imsave(nomfichier,imt)
    os.system(commande)


def viewimage_color(im,normalise=True,MINI=0.0, MAXI=255.0):
    """ Cette fonction fait afficher l'image EN COULEURS
        dans gimp. Si un gimp est deja ouvert il est utilise.
        Par defaut normalise=True. Et dans ce cas l'image est normalisee 
        entre 0 et 255 avant d'être sauvegardee.
        Si normalise=False MINI(defaut 0) et MAXI (defaut 255) seront mis a 0 et 255 dans l'image resultat
        
    """
    imt=np.float32(im.copy())
    if platform.system()=='Darwin': 
        prephrase='open -a GIMP '
        endphrase= ' '
    elif platform.system()=='Windows': 
        prephrase='"C:/Program Files/GIMP 2/bin/gimp-2.10.exe" '
        endphrase=' '
    else:
        prephrase='gimp '
        endphrase=' &'
    
    if normalise:
        m=imt.min()
        imt=imt-m
        M=imt.max()
        if M>0:
            imt=255*imt/M
    else:
        imt=(imt-MINI)/(MAXI-MINI)
        imt[imt<0]=0
        imt[imt>1]=1
        imt *= 255
    
    nomfichier=tempfile.mktemp('TPIMA.pgm')
    commande=prephrase +nomfichier+endphrase
    imt = imt.astype(np.uint8)
    skio.imsave(nomfichier,imt)
    os.system(commande)


def gray_world_correction(image):
    """
    Applique la correction de l'image basée sur l'hypothèse du monde gris.
    """  

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    R_mean = np.mean(R)
    G_mean = np.mean(G)
    B_mean = np.mean(B)

    mean_illuminant = (R_mean + G_mean + B_mean) / 3

    R = (R * (mean_illuminant / R_mean)).clip(0, 255)
    G = (G * (mean_illuminant / G_mean)).clip(0, 255)
    B = (B * (mean_illuminant / B_mean)).clip(0, 255)

    corrected_image = np.stack([R, G, B], axis=-1).astype(np.uint8)

    return corrected_image
