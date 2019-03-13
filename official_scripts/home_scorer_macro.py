#!/usr/bin/env python3
from __future__ import print_function
import json
import os.path
import sys
import math
from sklearn.metrics import f1_score
import numpy as np

# as per the metadata file, input and output directories are the arguments
[_, reference_file, submission_file] = sys.argv

submission = json.load(open(submission_file, 'r'))
truth_values = json.load(open(reference_file, 'r'))

#%%
# UNcomment if want to use this option  for macroF calculation
#def macro_f_score(y_true,y_pred):
#    
#    y_true = np.array(y_true)
#    y_pred = np.array(y_pred)
#    
#    tp = y_true == y_pred
#    tp_bins = y_true[tp]
#    tp_sum = np.bincount(tp_bins,minlength=3)
#    pred_sum = np.bincount(y_pred,minlength=3)
#    true_sum = np.bincount(y_true,minlength=3)
#    
#
#    precision = tp_sum/pred_sum 
#    precision[pred_sum==0] = 0.0
#
#    recall = tp_sum/true_sum
#    recall[true_sum==0] = 0.0
#    
#    f_score = (2 * precision * recall/(precision + recall))
#    f_score[tp_sum == 0] = 0.0
#    f_score = np.average(f_score)
#    
#    return f_score

def calculate_a_score(truth_values, submission):
 observed = 0.0
 correct = 0.0
 y_true = []
 y_pred = []
 total = len(truth_values.keys())

 for reference_id in truth_values.keys():
  if reference_id in submission.keys():
   observed += 1
   y_true.append(truth_values[reference_id])
   y_pred.append(submission[reference_id])
   if submission[reference_id] == truth_values[reference_id]:
    correct += 1
  else:
   print('unmatched entry:', reference_id, '-- no response for this reference')

 macroF = f1_score(y_true, y_pred, average='macro')
# macroF = macro_f_score(y_true,y_pred)  # either this option or the one above can be chosen, should return same
 return correct, total, macroF #, y_true, y_pred


def calculate_b_score(truth_values, submission):
#%%
 observed = 0
 correct = 0.0
 total = len(truth_values.keys())
 errors = []
 y_true = []
 y_pred = []
 for reference_id in truth_values.keys():
  if reference_id in submission.keys():
   observed += 1
   y_true.append(truth_values[reference_id])
   y_pred.append(submission[reference_id][0])
   try:
    yhat, confidence = submission[reference_id]
   except ValueError:
    print('   Each entry should be a list of two values - [veracity, confidence]')
    print('   veracity is one of "true" or "false"; confidence is a float from 0..1.')
    print('   This entry was:', submission[reference_id], ',  for document key', reference_id)
    sys.exit('-- error: data format: stopping')

   if yhat == truth_values[reference_id] and (yhat=="true" or yhat=="false"):
    correct += 1.0
    errors.append((1-confidence) ** 2)

   elif truth_values[reference_id] == 'unverified':
    errors.append((confidence) ** 2)	
   else:
    errors.append(1.0)
  else:
   print('unmatched entry:', reference_id, '-- no response for this reference')
   
 macroF = f1_score(y_true, y_pred, average='macro')
# macroF = macro_f_score(y_true,y_pred) 
 return correct, total, sum(errors), len(errors), macroF #, y_true, y_pred

#%%
acorrect = 0
atotal = 0
bcorrect = 0
btotal = 0
bsumerrors = 0.0
blenerrors = 0

englishascore = 0
englishamacro = 0
englishbscore = 0
englishbmacro = 0
englishbrmse = 0

danishascore = 0
danishamacro = 0
danishbscore = 0
danishbmacro = 0
danishbrmse = 0

russianascore = 0
russianamacro = 0
russianbscore = 0
russianbmacro = 0
russianbrmse = 0

macroa = 0
macrob = 0

#y_true_all_a = []
#y_pred_all_a = []
#
#y_true_all_b = []
#y_pred_all_b = []


fullsetpresent = 1
taskalangnum = 0
taskblangnum = 0

#%%
#ENGLISH

if('subtaskaenglish' in truth_values and len(truth_values['subtaskaenglish'])>0):
 taskalangnum+=1
 if('subtaskaenglish' in submission and len(submission['subtaskaenglish'])>0):
  correct, total, englishamacro = calculate_a_score(truth_values['subtaskaenglish'], submission['subtaskaenglish'])
  
  acorrect += correct
  atotal += total
  englishascore = correct/total
  print('Task A (SDQC), English, accuracy:', englishascore)
  print('Task A (SDQC), English, Macro averaged F1 score:', englishamacro)
  macroa = macroa + englishamacro
 else:
  print('No responses found for subtask A, English')
  fullsetpresent = 0
else:
 print('No truth data found for subtask A, English')

if('subtaskbenglish' in truth_values and len(truth_values['subtaskbenglish'])>0):
 taskblangnum+=1
 if('subtaskbenglish' in submission and len(submission['subtaskbenglish'])>0):
  correct, total, sumerrors, lenerrors, englishbmacro  = calculate_b_score(truth_values['subtaskbenglish'], submission['subtaskbenglish'])

  englishbscore = correct/total
  englishbrmse = 0
  if(lenerrors>0):
   englishbrmse = math.sqrt(sumerrors/lenerrors)
  bcorrect += correct
  btotal += total
  bsumerrors += sumerrors
  blenerrors += lenerrors
  print('Task B (veracity), English, accuracy:', englishbscore, "RMSE:", englishbrmse)
  print('Task B (veracity), English, Macro averaged F1 score:', englishbmacro)
  macrob = macrob+englishbmacro
 else:
  print('No responses found for subtask B, English')
  fullsetpresent = 0
else:
 print('No truth data found for subtask B, English')

#%%
#DANISH

if('subtaskadanish' in truth_values and len(truth_values['subtaskadanish'])>0):
 taskalangnum+=1
 if('subtaskadanish' in submission and len(submission['subtaskadanish'])>0):
  correct, total, danishamacro  = calculate_a_score(truth_values['subtaskadanish'], submission['subtaskadanish'])

  acorrect += correct
  atotal += total
  danishascore = correct/total
  print('Task A (SDQC), Danish, accuracy:', danishascore)
  print('Task A (SDQC), Danish, Macro averaged F1 score:', danishamacro)
  macroa = macroa + danishamacro
 else:
  print('No responses found for subtask A, Danish')
  fullsetpresent = 0
else:
 print('No truth data found for subtask A, Danish')

if('subtaskbdanish' in truth_values and len(truth_values['subtaskbdanish'])>0):
 taskblangnum+=1
 if('subtaskbdanish' in submission and len(submission['subtaskbdanish'])>0):
  correct, total, sumerrors, lenerrors, danishbmacro  = calculate_b_score(truth_values['subtaskbdanish'], submission['subtaskbdanish'])

  danishbscore = correct/total
  danishbrmse = 0
  if(lenerrors>0):
   danishbrmse = math.sqrt(sumerrors/lenerrors)
  bcorrect += correct
  btotal += total
  bsumerrors += sumerrors
  blenerrors += lenerrors
  print('Task B (veracity), Danish, accuracy:', danishbscore, "RMSE:", danishbrmse)
  print('Task B (veracity), Danish, Macro averaged F1 score:', danishbmacro)
  macrob = macrob+danishbmacro
 else:
  print('No responses found for subtask B, Danish')
  fullsetpresent = 0
else:
 print('No truth data found for subtask B, Danish')


#RUSSIAN

if('subtaskarussian' in truth_values and len(truth_values['subtaskarussian'])>0):
 taskalangnum+=1
 if('subtaskarussian' in submission and len(submission['subtaskarussian'])>0):
  correct, total, russianamacro = calculate_a_score(truth_values['subtaskarussian'], submission['subtaskarussian'])

  acorrect += correct
  atotal += total
  russianascore = correct/total
  print('Task A (SDQC), Russian, accuracy:', russianascore)
  print('Task A (SDQC), Russian, Macro averaged F1 score:', russianamacro)
  macroa = macroa + russianamacro
 else:
  print('No responses found for subtask A, Russian')
  fullsetpresent = 0
else:
 print('No truth data found for subtask A, Russian')

if('subtaskbrussian' in truth_values and len(truth_values['subtaskbrussian'])>0):
 taskblangnum+=1
 if('subtaskbrussian' in submission and len(submission['subtaskbrussian'])>0):
  correct, total, sumerrors, lenerrors, russianbmacro  = calculate_b_score(truth_values['subtaskbrussian'], submission['subtaskbrussian'])

  russianbscore = correct/total
  russianbrmse = 0
  if(lenerrors>0):
   russianbrmse = math.sqrt(sumerrors/lenerrors)
  bcorrect += correct
  btotal += total
  bsumerrors += sumerrors
  blenerrors += lenerrors
  print('Task B (veracity), Russian, accuracy:', russianbscore, "RMSE:", russianbrmse)
  print('Task B (veracity), Russian, Macro averaged F1 score:', russianbmacro)
  macrob = macrob+russianbmacro
 else:
  print('No responses found for subtask B, Russian')
  fullsetpresent = 0
else:
 print('No truth data found for subtask B, Russian')


ascore = 0
bscore = 0
brmse = 0
if(fullsetpresent==1 and atotal>0 and btotal>0):
 ascore = acorrect/atotal
 bscore = bcorrect/btotal
 brmse = 0
 if(blenerrors>0):
  brmse = math.sqrt(bsumerrors/blenerrors)


print('Task A (SDQC), accuracy:', ascore)

print('Task A (SDQC), MacroF (averaged over ' + str(taskalangnum) + ' languages):', macroa/taskalangnum)

print('Task B (veracity), accuracy:', bscore, "RMSE:", brmse)

print('Task B (veracity), MacroF (averaged over ' + str(taskblangnum) + ' languages):', macrob/taskblangnum)
