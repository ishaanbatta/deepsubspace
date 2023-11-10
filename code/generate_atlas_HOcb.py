import numpy as np 
import json 
import nibabel as nib
import pandas as pd


basepath = '/sysapps/ubuntu-applications/fsl/6.0.5.2/fsl/data/atlases/'

subcort = nib.load(basepath + 'HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz')
subcort_data = np.array(subcort.get_fdata(), dtype=int)
subcort_names = pd.read_csv('../../../data/atlases/HO-Subcort-ROIs.csv')
cort = nib.load(basepath + 'HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz')
cort_names = pd.read_csv('../../../data/atlases/HO-Cort-ROIs.csv')
cb = nib.load(basepath + 'Cerebellum/Cerebellum-MNIflirt-maxprob-thr25-1mm.nii.gz')
cb_data = np.array(cb.get_fdata(), dtype=int)
cb_names = pd.read_csv('../../../data/atlases/HO-CB-ROIs.csv')
# cb_data[cb_data>0] = 1 # Binarize, as CB is needed as one region
out_names = pd.DataFrame([],columns=cort_names.columns)
out_names = out_names.append(cort_names, ignore_index=True)
out_data = np.array(cort.get_fdata(), dtype=int)

# c1 = np.array([[i,(out_data==i).sum()] for i in np.unique(out_data) if i!=0 ])
# print(-np.sort(-c1[:,1]))
# c2 = np.array([[i,(subcort_data==i).sum()] for i in np.unique(subcort_data) if i!=0])
# print(-np.sort(-c2[:,1]))
# c3 = np.array([[i,(cb_data==i).sum()] for i in np.unique(cb_data) if i!=0])
# print(-np.sort(-c3[:,1]))

# Replace label numbers for subcortical data to append to cortical
offset = 48 # There are 0-47 labels in cortical regions
remove_indices = np.array([0,1,11,12])
for li in range(21):
    if li in remove_indices:
        print('Skipping %d'%li)
        continue
    out_data[subcort_data==(li+1)] = li+1+offset
    
# Remove cortical and white matter voxels from subcortical map
remove_values = remove_indices + 1 + offset
# remove_voxels = np.zeros(subcort_data.shape, dtype=bool)
adder = 0
new_map = {}
for v in range(1+offset,70):
    if v in remove_values:
        adder += 1
        out_data[out_data==v] = 0
        print('%d,%s will be removed'%(v, subcort_names['name'][v-offset-1]))
    else:
        print('Replacing %d,%s with %d,%s'%
              (v-adder,subcort_names['name'][v-offset-1-adder], v,subcort_names['name'][v-offset-1]))
        out_data[out_data==v] = v - adder
        names_row = {ci:subcort_names.iloc[v-offset-1][ci] for ci in subcort_names.columns}
        names_row['index'] = v - adder - 1
        out_names = out_names.append(names_row, ignore_index=True)
        new_map[v-1] = v-1 - adder  # -1 because of indices
    # remove_voxels = np.logical_or(remove_voxels, out_data == rv)
# out_data[np.where(remove_voxels==True)] = 0

# Add Cerbellar Regions with new indices for some of the combined regions.
cboffset = len(out_names)
new_cb_map = {}
for i in range(27):
    li = cb_names.iloc[i]['newindex']
    out_data[cb_data == i+1] = li+1 + cboffset
    
    if not li+cboffset in out_names['index']:
        ignore_cols = ['newindex','newname','newshortname']
        names_row = {ci:cb_names.iloc[i][ci] for ci in cb_names.columns if ci not in ignore_cols}
        names_row['index'] = cb_names.iloc[i]['newindex'] + cboffset
        names_row['shortname'] = cb_names.iloc[i]['newshortname']
        names_row['name'] = cb_names.iloc[i]['newname']    
        out_names = out_names.append(names_row, ignore_index=True)
    new_cb_map[i+offset] = i

outpath = '../../../data/atlases/'
nib.save(nib.Nifti1Image(np.array(out_data, dtype=int), affine=cort.affine, header=cort.header), outpath+'HO-CortSubcortCB-atlas.nii.gz')
out_names.to_csv(outpath+'HO-CortSubcortCB-ROIs.csv', index=False)
with open(outpath + 'new_subcort_roi_map.json','w+') as f:
    json.dump(new_map, f)
with open(outpath + 'new_CB_roi_map.json','w+') as f:
    json.dump(new_cb_map, f)