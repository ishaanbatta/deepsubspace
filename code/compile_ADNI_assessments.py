import numpy as np 
import pandas as pd
from scipy.io import loadmat
import glob
import os, sys
import pdb



adnimerge = pd.read_csv('/data/users2/ibatta/data/phenotype/ADNI/ADNIMERGE.csv')
with open('/data/users2/ibatta/data/phenotype/ADNI/ADNIMERGE_selected_column_names.txt') as fcols:
    select_columns = fcols.read().split('\n')[:-1]
adnimerge = adnimerge[select_columns]
adnimerge.rename(columns={'AGE':'age','DX':'ResearchGroup','PTGENDER':'Sex','PTID':'SubID','RAVLT_immediate':'RAVLT'}, inplace=True)
adnimerge['EXAMDATE'] = pd.to_datetime(adnimerge['EXAMDATE'],format='%Y-%m-%d')

research_groups = adnimerge['ResearchGroup'].unique()
research_groups = [ri for ri in research_groups if ri==ri] # Removes Nan values 




def to_npdatetime(str, format='%Y-%m-%d_%H_%M_%S.%f'):
    return np.datetime64(pd.to_datetime(str, format=format))




# #SMRI:

def extract_best_date_smri(subject_id, ref_date, findfile='smwc1T1', return_date=True):
    # raw_data_dir = '/data/qneuromark/Data/ADNI/Updated/T1/ADNI/'
    raw_data_dir = '/data/users2/ibatta/data/features/lowresSMRI/ADNI/nii/'
    keywords = ['MPRAGE','MP-RAGE','MP_RAGE','MPRAGE_Repeat','MP-RAGE_Repeat','MP_RAGE_Repeat','MPRAGE_*','MP-RAGE*','MP_RAGE*','*Accelerated*','*']
    
    candidate_dirs = []
    candidate_files = []
    for kw in keywords:
        dirpath = raw_data_dir + '/' + subject_id + '/' + kw + '/*/*/anat/%s.nii*'%findfile
        candidate_files += glob.glob(dirpath)
        # print('candidate_files for %s: '%dirpath)
        # print(candidate_files)
            
        # if len(candidate_files) > 0:
        #     # candidate_date_dirs = [glob.glob(dr+'/*') for dr in candidate_dirs]
        #     # date_dir_lengths = [len(dr) for dr in candidate_date_dirs]
        #     # final_date_dirs = candidate_date_dirs[np.argmax(date_dir_lengths)]
        #     # print('candidate_date_dirs:')
        #     # print(candidate_date_dirs)
        #     # print('final_date_dir:')
        #     # print(final_date_dirs)
        #     break
    
    # pdb.set_trace()
    final_dates = np.array([dr.split('/')[-4] for dr in candidate_files])
    final_dates = [to_npdatetime(dt) for dt in final_dates]
    final_dates = np.array(final_dates, dtype=np.datetime64)
    date_diffs = np.abs(final_dates-ref_date)
    mindiff_index = np.argmin(date_diffs)

    final_nii_file = candidate_files[mindiff_index]
    mindate_diff = pd.to_timedelta(date_diffs[mindiff_index])
    if not return_date:
        return final_nii_file
    else:
        return final_nii_file, final_dates[mindiff_index], mindate_diff

# ## Creates analysis_SCORE files from ADNI_merge file for a given list of structural subject_ids.

# with open('/data/users2/ibatta/data/features/SFNC/ADNI/subject_ids.txt') as fsubs:
# with open('/data/users2/ibatta/data/phenotype/ADNI/m1l1t1u_MMR180d.txt') as fsubs:
with open('/data/users2/ibatta/data/features/lowresSMRI/ADNI/SMRI_analysis_score_uniq_subjects.txt') as fsubs:
    sfnc_subject_ids = fsubs.read().split('\n')[:-1]
# sfnc_dates = np.loadtxt('/data/users2/ibatta/data/features/fmrimeasures/ADNI/dates_MMR180d.csv', dtype=np.datetime64)
sfnc_dates = np.loadtxt('/data/users2/ibatta/data/features/lowresSMRI/ADNI/SMRI_analysis_score_uniq_dates.txt', dtype=np.datetime64)

# with open('/data/users2/ibatta/data/features/SMRI/ADNI/subject_ids.txt') as fsubs:
    # subject_ids = fsubs.read().split('\n')[:-1]
subject_ids = sfnc_subject_ids

raw_data_dir = '/data/users2/ibatta/data/features/lowresSMRI/ADNI/nii'

raw_files_to_be_used = []
smri_dates = []
smri_datediffs = []
smri_diff_threshold = 180


df_final_merge = pd.DataFrame([],columns=adnimerge.columns)

for sindex in range(len(subject_ids)):
    si = subject_ids[sindex]

    sdf_merge = adnimerge.loc[adnimerge['SubID']==si]
    dates_merge = sdf_merge['EXAMDATE'].values

    if si in sfnc_subject_ids:
        sindex_sfnc = sfnc_subject_ids.index(si)
        earliest_date = sfnc_dates[sindex_sfnc]
        mindate_index = np.argmin(np.abs(dates_merge - earliest_date))
    else:
        mindate_index = np.argmin(dates_merge)
        earliest_date = dates_merge[mindate_index]
    
    smrifile, smridate, smridiff = extract_best_date_smri(si, earliest_date)
    numdays = int(str(smridiff).split('days')[0].strip(' '))
    if numdays > 180:
        ('Ignoring Subject %s: fMRI Scan Date and sMRI scan dates are %d (more than %d) days apart!'%(si,numdays, smri_diff_threshold))
        continue 
    df_final_merge = df_final_merge.append(sdf_merge.iloc[mindate_index])
    raw_files_to_be_used.append(smrifile)
    smri_dates.append(smridate)
    smri_datediffs.append(str(smridiff))

    
with open('/data/users2/ibatta/data/features/lowresSMRI/ADNI/filelist_l1t1u180d.txt','w') as fl:
    fl.write('\n'.join(raw_files_to_be_used)+'\n')
with open('/data/users2/ibatta/data/features/lowresSMRI/ADNI/SMRI_datediffs_l1t1u180d.txt','w') as fl:
    fl.write('\n'.join(smri_datediffs)+'\n')

np.savetxt('/data/users2/ibatta/data/features/lowresSMRI/ADNI/dates_l1t1u180d.csv', smri_dates, delimiter=',', fmt='%s')

df_final_merge.insert(len(df_final_merge.columns),"lowres_smriPath",raw_files_to_be_used)
df_final_merge.to_csv('/data/users2/ibatta/data/phenotype/ADNI/analysis_SCORE_SMRI_l1t1u180d.csv', index=False)



sys.exit(0)



# #SFNC



## Creates analysis_SCORE files from ADNI_merge file for a given list of functional subject_ids.

# with open('/data/users2/ibatta/data/features/SFNC/ADNI/subject_ids.txt') as fsubs:
with open('/data/users2/ibatta/data/phenotype/ADNI/m1l1t1u_MMR180d.txt') as fsubs:
    subject_ids = fsubs.read().split('\n')[:-1]

dfadni = pd.read_csv('/data/users2/ibatta/data/phenotype/ADNI/Functional/ADNI_demos_measures.csv')

df_final_merge = pd.DataFrame([],columns=adnimerge.columns)
sfnc_data = []
sfnc_paths = []
sfnc_dates = []
date_diffs = []
mindiff_threshold = 180

# _merge refers to phenotypes extracted from ADNI merge file, else they belong ADNI_demos file.
for sindex in range(len(subject_ids)):
    si = subject_ids[sindex]

    sdf = dfadni.loc[dfadni['SubID'] == si] 
    fpdf = sdf['fdir'].str.split('/',expand=True)
    fpdf2 = sdf['fdir'].str.split('/',expand=True)
    fpdf[11] = pd.to_datetime(fpdf[11], format='%Y-%m-%d_%H_%M_%S.%f') ## 12th field is date-time
    sdf_merge = adnimerge.loc[adnimerge['SubID']==si]

    if np.any(fpdf[10]=='Resting_State_fMRI'):
        fpdf = fpdf[fpdf[10]=='Resting_State_fMRI']
    fpdf = fpdf.sort_values(by=11)

    if (len(fpdf) == 0) or (len(sdf_merge)==0):
        print('Data not found for '+si)
    else:
        # sfc_filename = sdf.loc[fpdf.index[0]]['fc_dir'] + '/sfc.mat'
        # sfnc_data.append(loadmat(sfc_filename)['sFNC'].squeeze())

        earliest_date = fpdf[11].values[0]
        
        dates_merge = sdf_merge['EXAMDATE'].values
        mindiff_index = np.argmin(np.abs(dates_merge - earliest_date))
        mindiff = np.min(np.abs(dates_merge - earliest_date))
        mindiff = pd.to_timedelta(mindiff)
        
        numdays = int(str(mindiff).split('days')[0].strip(' '))
        if numdays > mindiff_threshold:
            print('Ignoring Subject %s: fMRI Scan Date and Clinical Assessment Date are %d (more than %d) days apart!'%(si,numdays, mindiff_threshold))
            continue
        if sdf_merge['ResearchGroup'].values[mindiff_index]  not in research_groups:
            print('Ignoring Subject %s: No Label available'%si)
            continue
        sfnc_dates.append(earliest_date)
        date_diffs.append(mindiff)
        sfnc_paths.append('/'.join(fpdf2.loc[fpdf.index[0]].values))
        selected_merge_date = dates_merge[mindiff_index]
        df_final_merge = df_final_merge.append(sdf_merge.iloc[mindiff_index])
        

sfnc_data = np.array(sfnc_data)
# np.savetxt('/data/users2/ibatta/data/features/SFNC/ADNI/SFNCflat.csv', sfnc_data, delimiter=',')
np.savetxt('/data/users2/ibatta/data/features/fmrimeasures/ADNI/dates_MMR%dd.csv'%mindiff_threshold, sfnc_dates, delimiter=',', fmt='%s')
np.savetxt('/data/users2/ibatta/data/features/fmrimeasures/ADNI/date_diffs_MMR%dd.csv'%mindiff_threshold,date_diffs,fmt='%s')
df_final_merge.insert(len(df_final_merge.columns),"lowres_fmriPath",sfnc_paths)
df_final_merge.to_csv('/data/users2/ibatta/data/phenotype/ADNI/analysis_SCORE_MMR%dd.csv'%mindiff_threshold, index=False)



