# mathematical_modeling

## 2021全国大学生数学建模大赛C题 摘要 
 
​		产企业原材料的订购与运输方案是生产成本控制所研究的重要课题。本文应用主流的 TOPSIS 评价模型和自定义的缺失评价模型，从历史订单和供货数据中给予不同供应商合理的评价，并 根据评价指标选择出重要性最高的若干供应商。进一步地构造出多种基于时间序列分析法的供应 商供货量预测模型、转运商运输方案决策模型，保障了企业在原材料订购和运输方面的最优成本 控制，并为企业在现有条件下的原材料供给上限做出了合理预测，为企业提高自身产能等进一步 发展提供了合理指导。

​		针对问题一，为研究不同供货商对于企业的重要性，通过建立基于客观数据分析的" 缺失评价 模型"，即通过人工干预的方法去除某家供货商在历史上对于企业的供货量，计算该供货商供货量 的缺失对企业产能造成的损失作为该供货商重要性评价的指标，并为总共 402 家供货商按照指标 降序的方式排序，筛选出对企业最重要的 50 家供货商。同时建立主流 TOPSIS 评价模型对 402 家 供货商进行评价，同样选出 50 家重要的供货商，用于和" 缺失评价模型" 的结果进行比对，最终发 现二者选出的供货商仅有一家不同，说明本文建立的缺失评价模型在具有创新性的同时兼具了高 度的可靠性。 

​		针对问题二，根据附件 1 中的历史数据，通过构建长度为 24 周的滑动窗口模型，筛选出符合 后续预测问题需要的可靠数据区间，通过 0-1 规划模型计算出最少需要 25 家供货商满足企业的产 能，回答了第一小问。之后基于时间序列分析法和整数规划法构建了在该可靠数据区间上的供应 商供货量预测模型，并根据概率论知识构建供货量和订单量的转换模型，为企业制定出了合理且 最优的 24 周原材料采购方案，回答了第二小问。之后同样基于时间序列分析法对不同转运商的运 损率进行分析，通过整数规划法构建转运商运输方案的决策模型，回答了第三小问。通过上述模 型制定出的采购方案和运输方案效果均达到理论预期，具体方案详见附件 A，附件 B。 

​		针对问题三，需要在问题二中构建的基于时间序列分析的规划模型中，增加题干中要求的多 采购 A 类材料且少采购 C 类材料的约束条件，其他求解均可参照问题二。具体方案详见附件 A， 附件 B。 

​		针对问题四，通过上述问题二、三建立的分析模型，对供货商和转运商进行分析发现，在当前条 件下该企业每周所能入库的原材料的最大值受限于 8 家转运商的最高运力，即 6000×8 = 48000𝑚 3， 根据此约束条件，运用整数规划模型，制定了相应的原材料采购模型和转运模型，进一步求解出 相关方案，详见附件 A，附件 B。计算出该企业每周产能的上限为 35747.06𝑚 3，相较于之前可以 提高 26.76%。 

​		综上所述，本文在多种不同尺度的约束条件下，均构建出基于时间序列分析的订购和转运方 案制定模型，并求出相应约束条件下生产成本最低的方案。该套模型具备极优的规划能力和极强 的适应性，可以将其推广至面向实体制造的工业界，并为企业估测自身产能上限，制定发展 战略提供了重要的参考价值。 



**关键词**: TOPSIS 评价模型 时间序列分析法 Python 线性规划 概率论 机器学习
 
  
提交时间 21:47:42 P.M.  
2021年9月12日（星期五） (GMT+8)  
