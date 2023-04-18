"""
Copyright 2018 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.stats import ttest_ind, ttest_1samp
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns 
import pandas as pd
# /zhome/94/5/127021/speciale/tcav/tcav/utils_plot.py
# helper function to output plot and write summary data
def plot_results(results, random_counterpart=None, random_concepts=None, num_random_exp=100,
    min_p_val=0.05, alternative = 'two-sided', t_test_mean = None, bonferroni_nr = None,
    plot_hist = False, save_fig = False, nameOfDataFrame = 'results'):
  """Helper function to organize results.
  When run in a notebook, outputs a matplotlib bar plot of the
  TCAV scores for all bottlenecks for each concept, replacing the
  bars with asterisks when the TCAV score is not statistically significant.
  If you ran TCAV with a random_counterpart, supply it here, otherwise supply random_concepts.
  If you get unexpected output, make sure you are using the correct keywords.
  Args:
    results: dictionary of results from TCAV runs.
    random_counterpart: name of the random_counterpart used, if it was used. 
    random_concepts: list of random experiments that were run. 
    num_random_exp: number of random experiments that were run.
    min_p_val: minimum p value for statistical significance
    t_test_mean: if value is given a 1-sample t-test will beformed
    plot_hist: to plot histograms of results 
    alternative: input to t-test 
  """
  font_title = {#'family': 'serif',
        #'color':  'darkgrey',
        'weight': 'normal',
        'size': 16,
        }
  font_axis = {#'family': 'serif',
        #'color':  'darkgrey',
        'weight': 'normal',
        'size': 12,
        }
  font_leg = {#'family': 'serif',
        'color':  'darkgrey',
        'weight': 'normal',
        'size': 8,
        }
  # print('P-values:',min_p_val)

  # helper function, returns if this is a random concept
  def is_random_concept(concept):
    if random_counterpart:
      return random_counterpart == concept
    
    elif random_concepts:
      return concept in random_concepts

    else:
      print('>>> is random concept:', concept)
      return 'random500_' in concept

  # print class, it will be the same for all
  # print("Class =", results[0]['target_class'])

  if bonferroni_nr == None:
    bonferroni_nr = 1

  min_p_val = min_p_val/bonferroni_nr
  # prepare data
  # dict with keys of concepts containing dict with bottlenecks
  result_summary = {}

  # random
  random_i_ups = {}
    
  for result in results:
    if result['cav_concept'] not in result_summary:
      result_summary[result['cav_concept']] = {}

    if result['bottleneck'] not in result_summary[result['cav_concept']]:
      result_summary[result['cav_concept']][result['bottleneck']] = []
    
    result_summary[result['cav_concept']][result['bottleneck']].append(result)

    # store random
    if is_random_concept(result['cav_concept']):
      if result['bottleneck'] not in random_i_ups:
        random_i_ups[result['bottleneck']] = []
        
      random_i_ups[result['bottleneck']].append(result['i_up'])
    
  # to plot, must massage data again 
  plot_data = {}
  plot_concepts = []
  count = 0  
  df_result = pd.DataFrame(columns = ['TCAV score','Concept','Bottleneck'])
  # print concepts and classes with indentation
  for concept in result_summary:
        
    # if not random
    if not is_random_concept(concept):
      # print(" ", "Concept =", concept)
      plot_concepts.append(concept)

      for bottleneck in result_summary[concept]:
        i_ups = [item['i_up'] 
                 for item in result_summary[concept][bottleneck]]
        
        if bottleneck not in df_result['Bottleneck'].unique():
          df_random = pd.DataFrame(random_i_ups[bottleneck],columns=['TCAV score'])
          df_random['Concept'] = 'random'
          df_random['Bottleneck'] = bottleneck
          df_result = pd.concat([df_result, df_random], ignore_index=True)

        df_concept = pd.DataFrame(i_ups,columns=['TCAV score'])
        df_concept['Concept'] = concept
        df_concept['Bottleneck'] = bottleneck
        df_result = pd.concat([df_result, df_concept], ignore_index=True)

        # Calculate statistical significance
        if t_test_mean == None:
          _, p_val = ttest_ind(random_i_ups[bottleneck], i_ups)#, alternative = alternative)

        else:
          _, p_val = ttest_1samp(i_ups, t_test_mean)#,  alternative = alternative)
          _, p_val_random = ttest_1samp(random_i_ups[bottleneck], t_test_mean)#,  alternative = alternative)
        
        if count == 0:
          print('>>> Number of TCAV concept observations <<<\n', len(i_ups))
          print('>>> Number of TCAV random observations <<<\n', len(random_i_ups[bottleneck]))
          count = 1

        if bottleneck not in plot_data:
          plot_data[bottleneck] = {'random_p-value':[], 'bn_vals': [], 'bn_stds': [], 'significant': [], 'p-value': [], 'concept':[]}
          plot_data[bottleneck]['random_p-value'].append(np.mean(random_i_ups[bottleneck]))
          plot_data[bottleneck]['random_p-value'].append(np.std(random_i_ups[bottleneck]))
          if t_test_mean != None:  
            plot_data[bottleneck]['random_p-value'].append(p_val_random)
            if p_val_random > min_p_val:
              plot_data[bottleneck]['random_p-value'].append('False')
            else:
              plot_data[bottleneck]['random_p-value'].append('True')

        if p_val > min_p_val:
          # statistically insignificant
          plot_data[bottleneck]['bn_vals'].append(0.01)
          plot_data[bottleneck]['bn_stds'].append(0)
          plot_data[bottleneck]['significant'].append(False)
          plot_data[bottleneck]['p-value'].append(p_val)
          plot_data[bottleneck]['concept'].append(concept)
            
        else:
          plot_data[bottleneck]['bn_vals'].append(np.mean(i_ups))
          plot_data[bottleneck]['bn_stds'].append(np.std(i_ups))
          plot_data[bottleneck]['significant'].append(True)
          plot_data[bottleneck]['p-value'].append(p_val)
          plot_data[bottleneck]['concept'].append(concept)

        """
        print(3 * " ", "Bottleneck =", ("%s. TCAV Score = %.2f (+- %.2f), "
            "random was %.2f (+- %.2f). p-val = %.3f (%s)") % (
            bottleneck, np.mean(i_ups), np.std(i_ups),
            np.mean(random_i_ups[bottleneck]),
            np.std(random_i_ups[bottleneck]), p_val,
            "not significant" if p_val > min_p_val else "significant"))
        """

    compression_opts = dict(method='zip',
                        archive_name=nameOfDataFrame + '.csv') 

    df_result.to_csv(nameOfDataFrame + '.zip', index=False, #BREYTA
         compression=compression_opts) 

  # histogram plots
  # Moved this to collab
  # 'if plot_hist:

  #   palette ={"left_MI": "blue", "right_MI": "red", "eyem": "g", "random": "grey", "musc" : "yellow"}
    
  #   for bottlenecks in df_result['Bottleneck'].unique():
  #     data = df_result[df_result['Bottleneck'] == bottlenecks]
  #     # set figure
  #     plt.subplots(nrows=1, ncols=4, sharey=True,figsize=(15,4));
  #     #plt.suptitle(f'Histogram of TCAV scores for each concept in {bottlenecks}', fontsize =20);
      
  #     # first concept
  #     plt.subplot(1, 4, 1);
  #     ax = sns.histplot(data=data[data['Concept'].isin(['left_MI','random'])], x="TCAV score", hue_order =['left_MI','random'],
  #     hue="Concept", stat = 'frequency', binrange = (0,1),common_norm=False, bins = 20, element="step", palette=palette);
  #     sns.move_legend( ax, loc = "upper left", fontsize = 'large');
  #     ax.set_xlabel("TCAV score")#,fontsize = 'xx-large');
  #     ax.set_ylabel("Frequency")#,fontsize =  'xx-large');
  #     plt.axvline(0.5, 0,10, ls = '--', lw = 0.8, color = 'grey');
  #     # 2nd
  #     plt.subplot(1, 4, 2);
  #     ax = sns.histplot(data=data[data['Concept'].isin(['right_MI','random'])], x="TCAV score", hue_order = ['right_MI','random'],
  #     hue="Concept", stat = 'frequency', binrange = (0,1),common_norm=False, bins = 20, element="step", palette=palette);
  #     sns.move_legend( ax, loc = "upper left",fontsize = 'large');
  #     ax.set_xlabel("TCAV score")#,fontsize = 'xx-large');
  #     plt.axvline(0.5, 0,10, ls = '--', lw = 0.8, color = 'grey');
  #     # 3rd
  #     plt.subplot(1, 4, 3);
  #     ax = sns.histplot(data=data[data['Concept'].isin(['eyem','random'])], x="TCAV score", hue="Concept",
  #     hue_order = ['eyem','random'],stat = 'frequency', binrange = (0,1),common_norm=False, bins = 20, element="step", palette=palette);
  #     sns.move_legend( ax, loc = "upper left", fontsize = 'large');
  #     ax.set_xlabel("TCAV score")#,fontsize = 'xx-large');
  #     plt.axvline(0.5, 0,10, ls = '--', lw = 0.8, color = 'grey');

  #     plt.subplot(1, 4, 4);
  #     ax = sns.histplot(data=data[data['Concept'].isin(['musc','random'])], x="TCAV score", hue="Concept",
  #     hue_order = ['musc','random'],stat = 'frequency', binrange = (0,1),common_norm=False, bins = 20, element="step", palette=palette);
  #     sns.move_legend( ax, loc = "upper left", fontsize = 'large');
  #     ax.set_xlabel("TCAV score")#,fontsize = 'xx-large');
  #     plt.axvline(0.5, 0,10, ls = '--', lw = 0.8, color = 'grey');
      
  #     # finish figure
  #     plt.tight_layout();
  #     if save_fig:
  #       print('Now overwritting and saving figure')
  #       plt.savefig(f'./BENDR/SavedResults_tcav/histogram_{results[0]["target_class"]}_{bottlenecks}.pdf')
  #     plt.show();'
      
      
  # subtract number of random experiments
  if random_counterpart:
    num_concepts = len(result_summary) - 1
  elif random_concepts:
    num_concepts = len(result_summary) - len(random_concepts)
  else: 
    num_concepts = len(result_summary) - num_random_exp
    
  num_bottlenecks = len(plot_data)
  bar_width = 0.35
    
  # create location for each bar. scale by an appropriate factor to ensure 
  # the final plot doesn't have any parts overlapping
  index = np.arange(num_concepts) * bar_width * (num_bottlenecks + 1)

  # matplotlib
  
  fig, ax = plt.subplots(figsize = (10,6))
  # draw all bottlenecks individually
  for i, [bn, vals] in enumerate(plot_data.items()):
    bar = ax.bar(index + i * bar_width, vals['bn_vals'],
        bar_width, yerr=vals['bn_stds'],ecolor = 'grey', label=bn, color = sns.color_palette("Paired")[i])
    # draw stars to mark bars that are stastically insignificant to 
    # show them as different from others
    for j, significant in enumerate(vals['significant']):
      if not significant:
        ax.text(index[j] + i * bar_width - 0.1, 0.01, "*",
            fontdict = {'weight': 'bold', 'size': 16,
            'color': bar.patches[0].get_facecolor()})
  # set properties
  #ax.set_title('TCAV Scores for each concept and bottleneck', fontsize = 20)
  ax.set_ylabel('TCAV Score', fontsize = 'large')
  ax.set_xticks(index + num_bottlenecks * bar_width / 2)
  ax.set_xticklabels(plot_concepts, fontsize = 'large')#fontsize = 16)
  ax.legend(fontsize = 'large')
  fig.tight_layout()
  if save_fig:
    print('Now overwritting and saving figure')
    plt.savefig(f'./BENDR/SavedResults_tcav/barplot_{results[0]["target_class"]}_bonferroni_{bonferroni_nr}.pdf')

  # ct stores current time
  # ct = datetime.datetime.now()
  # plt.savefig(f'SavedResults/results_{ct}.png')

  return plot_data
