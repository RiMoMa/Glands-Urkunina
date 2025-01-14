# batch generator by MIC-DKFZ: https://github.com/MIC-DKFZ/batchgenerators
import numpy as np
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
import openslide
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
import subprocess


class DataLoader(DataLoader):
    def __init__(self, data, svs_path,ImgNormalize,CaseName, batch_size, patch_size, num_threads_in_multithreaded, crop_status=False, crop_type="center", seed_for_shuffle=1234, return_incomplete=False, shuffle=True, infinite=True, margins=(0,0,0),inputDataset=None):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the returned batch will have
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        # original patch size with [width, height]
        self.patch_size = patch_size
        self.n_channel = 3
        self.indices = list(range(len(data)))
        self.crop_status = crop_status
        self.crop_type = crop_type
        self.margins = margins
        self.svs_path = svs_path
        self.ImgNormalize = ImgNormalize
        self.CaseName = CaseName
        self.inputDataset = inputDataset
    @staticmethod
    def load_patient(imgSVS,x,y,patchsize):
        img = imgSVS.read_region((x, y), 0, (patchsize, patchsize))
        img = np.array(img)[:,:,0:3]
        img = Image.fromarray(img)
        img = img.resize((512, 512),Image.BILINEAR)
        return img


    @staticmethod
    def save_patient(imgSVS,x,y,patchsize,name,PathDataset):
        img = imgSVS.read_region((x, y), 0, (patchsize, patchsize))
        img = np.array(img)[:,:,0:3]
        is_white = np.sum(img.ravel() > 200) / len(img.ravel()) > 0.80

        if not is_white:
            img = Image.fromarray(img)
            # save a
            ImgPathSave = PathDataset + 'img_X_' + str(x) + '_Y_' + str(y) + '_' + str(name[0]) + '_' + str(
                name[1]) + '.png'
            img.save(ImgPathSave)

#        img = Image.fromarray(img)
        #save a
 #       ImgPathSave = PathDataset+'img_X_'+str(x)+'_Y_'+str(y)+'_'+str(name[0])+'_'+str(name[1])+'.png'
  #      img.save(ImgPathSave )
        #img = img.resize((512, 512),Image.BILINEAR)
        return 0

    @staticmethod
    def load_patient_imNormalized(imgSVS,x,y,patchsize,name,PathDataset):
        ImgPathSave = PathDataset + 'img_X_' + str(x) + '_Y_' + str(y) + '_' + str(name[0]) + '_' + str(name[1]) + '.png'
        img = Image.open(ImgPathSave)
        img = img.resize((512, 512),Image.BILINEAR)
        return img

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        CaseName = self.CaseName
        slide = openslide.OpenSlide(self.svs_path)

        idx = self.get_indices()
        gland_img = [list(self._data.items())[i] for i in idx]
        # initialize empty array for data and seg
        img = np.zeros((len(gland_img), self.n_channel, 512,512), dtype=np.float32)
        # iterate over patients_for_batch and include them in the batch
        #extraction_coords = data
        tileInfo = []
        tileCoords = []

        if self.ImgNormalize==False:

            for i,(nombre, (x, y)) in enumerate(gland_img):
                tileInfo.append(nombre)
                tileCoords.append((x, y))
                img_data = self.load_patient(slide, x, y,self.patch_size[0])
                # hence we use tensor manipulation to convert to channel first
                # dummy dimension in order for it to work (@Todo, could be improved)
                img_data = np.einsum('hwc->chw', img_data)
                if self.crop_status:
                    img_data, seg_data = crop(img_data[None], seg=seg_data[None], crop_size=self.patch_size,
                                        margins=self.margins, crop_type=self.crop_type)
                    img[i] = img_data[0]
                else:
                    img[i] = img_data
        else:
            PathDataset = 'Dataset_no_Normalized/' + CaseName + '/'
            os.makedirs(PathDataset, exist_ok=True)
            # for i, (nombre, (x, y)) in enumerate(gland_img):
            #     self.save_patient(slide, x, y, self.patch_size[0],nombre,PathDataset )
            # MAX_WORKERS = 12
            # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            #     for i, (nombre, (x, y)) in enumerate(gland_img):
            #         executor.submit(self.save_patient, slide, x, y, self.patch_size[0], nombre, PathDataset)

            #correr normalizacionMatlab
            #NormalizacionColor_registro('DatasetNormalized/U001/*.png', 'DatasetsiNormalized/U001/')
            import subprocess

            # Define las variables de entrada y salida
            input_path = PathDataset+'*.png' #Todo: Aca se debe poner la entra del folder de salida y  no la base del script
            output_path = self.inputDataset+'/Dataset_Normalized/' + CaseName + '/'
            #
            # # Construye el comando
            # command = (
            #     f"addpath('/home/ricardo/Projects/Gland-Segmentation/'); "
            #     f"NormalizacionColor_registro('{input_path}', '{output_path}'); exit;"
            # )
            #
            # # Llama a MATLAB con subprocess.call
            # subprocess.call(
            #     [
            #         '/usr/local/MATLAB/R2024a/bin/matlab',
            #         '-nodisplay',
            #         '-nosplash',
            #         '-r',
            #         command
            #     ]#,
            #     #stdout=subprocess.DEVNULL,
            #     #stderr=subprocess.DEVNULL
            # )
            i=0
            for x, (nombre, (x, y)) in enumerate(gland_img):

                if os.path.exists(output_path + 'img_X_' + str(x) + '_Y_' + str(y) + '_' + str(nombre[0]) + '_' + str(nombre[1]) + '.png'):

                    img_data = self.load_patient_imNormalized(slide, x, y, self.patch_size, nombre, output_path)
                    tileInfo.append(nombre)
                    tileCoords.append((x, y))


                    # hence we use tensor manipulation to convert to channel first
                    # dummy dimension in order for it to work (@Todo, could be improved)
                    img_data = np.einsum('hwc->chw', img_data)
                    if self.crop_status:
                        img_data, seg_data = crop(img_data[None], seg=seg_data[None], crop_size=self.patch_size,
                                                  margins=self.margins, crop_type=self.crop_type)
                        img[i] = img_data[0]
                    else:
                        img[i] = img_data
                    i = i + 1
                else:
                    continue
        img = np.delete(img, slice(i, img.shape[0]), axis=0)
        return {'data': img,'tileInfo':tileInfo,'tileCoords':tileCoords}

