/*
 *    NNDVI.java
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *    
 */
package moa.classifiers.meta;

import java.util.ArrayList;
import java.util.Collections;

import org.apache.commons.math3.distribution.NormalDistribution;

import weka.core.Instances;
import weka.core.neighboursearch.KDTree;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddID;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;
import moa.options.ClassOption;

/**
 * Nearest Neighbor-based Density Variation Identification (NN-DVI)
 * Please cite Accumulating regional density dissimilarity for concept drift detection
 * <p>
 */
public class NNDVI extends AbstractClassifier implements MultiClassClassifier {

	private static final long serialVersionUID = 1L;

	public IntOption minWOption = new IntOption("minW", 'W', "min drift detection window size", 100, 50,
			Integer.MAX_VALUE);
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class,
			"bayes.NaiveBayes");
	public IntOption kNNPSOption = new IntOption("kNNPS", 'K',
			"The number of neighbors for constructing NNPS, set 0: kNNPS=max{independence}, default kNNPS=30 for minW=100",
			30, 0, Integer.MAX_VALUE);

	public IntOption samplingOption = new IntOption("s", 'S', "Sampling times", 50, 50, Integer.MAX_VALUE);
	public FloatOption ASLOption = new FloatOption("alpha", 'A', "significance level", 0.01, 0, 1);
	public IntOption maxBufferOption = new IntOption("maxBuff", 'B', "max buffer size", 1000, 200, Integer.MAX_VALUE);

	private WekaToSamoaInstanceConverter w2s = new WekaToSamoaInstanceConverter();
	private SamoaToWekaInstanceConverter s2w = new SamoaToWekaInstanceConverter();

	protected int minW = minWOption.getValue();
	protected Classifier baseLearner;
	protected int kNNPS = kNNPSOption.getValue();
	protected int sTimes = samplingOption.getValue();
	protected double ASL = ASLOption.getValue();
	protected int maxBuff = maxBufferOption.getValue();
	protected Instances knowledgeBase = null;
	protected Instances oldWin = null;
	protected Instances curWin = null;
	protected double driftThreshold = -1;

	/**
	 * Determines whether the classifier is randomizable.
	 */
	public boolean isRandomizable() {
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {

		baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
		baseLearner.resetLearning();
		if (knowledgeBase == null) {
			baseLearner.trainOnInstance(inst);

		} else {
			for (weka.core.Instance _i : knowledgeBase) {
				baseLearner.trainOnInstance(w2s.samoaInstance(_i));
			}
		}

		return baseLearner.getVotesForInstance(inst);
	}

	@Override
	public void resetLearningImpl() {

		minW = minWOption.getValue();
		kNNPS = kNNPSOption.getValue();
		sTimes = samplingOption.getValue();
		ASL = ASLOption.getValue();
		maxBuff = maxBufferOption.getValue();
		knowledgeBase = null;
		oldWin = null;
		curWin = null;
		driftThreshold = -1;

		baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
		baseLearner.resetLearning();

	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
	

		if (oldWin == null) {
			oldWin = new Instances(s2w.wekaInstance(inst).dataset(), 0, 0);
			knowledgeBase = new Instances(oldWin);
		}

		if (knowledgeBase.numInstances() >= maxBuff) {
			knowledgeBase.delete(0);
		}

		if (oldWin.numInstances() < minW) {
			oldWin.add(s2w.wekaInstance(inst));
			knowledgeBase.add(s2w.wekaInstance(inst));
		} else {
			/*
			 * detect drift between the first batch and the last batch shared
			 * instances are allowed
			 * 
			 * batch1 = [d1, ..., di] batch2 = [dj, ..., dn] batch1 union batch2
			 * = kb
			 */
			boolean driftFlag = false;

			if (curWin == null || curWin.numInstances() == 0) {
				curWin = new Instances(oldWin);
			}

			curWin.add(s2w.wekaInstance(inst));

			curWin.delete(0);

			try {

				driftFlag = driftDetection(oldWin, curWin);
				if (driftFlag) {
					System.out.println("		drift: " + driftFlag);
				}
				driftThreshold = -1;
			} catch (Exception e1) {
				e1.printStackTrace();
			}

			if (driftFlag) {

				oldWin = new Instances(curWin);
				knowledgeBase = new Instances(oldWin);

				curWin.clear();

			} else {
				knowledgeBase.add(s2w.wekaInstance(inst));

			}
		}

	}

	protected boolean driftDetection(Instances _histBatch, Instances _newBatch) throws Exception {

		Instances hB = new Instances(_histBatch);
		Instances nB = new Instances(_newBatch);

		ArrayList<Double> label = new ArrayList<Double>();

		for (weka.core.Instance _i : hB) {
			label.add(_i.classValue());
		}
		for (weka.core.Instance _i : nB) {
			label.add(_i.classValue());
		}

		int hBsize = hB.size();
		int nBsize = nB.size();

		double[][] nnps = buildNNPS_Weight(hB, nB, kNNPS);

		ArrayList<Integer> shuffleIdx = new ArrayList<Integer>();
		for (int i = 0; i < hBsize + nBsize; i++) {
			shuffleIdx.add(i);
		}

		double actDist = Math.log(getNNPSjd(nnps, new ArrayList<Integer>(shuffleIdx.subList(0, hBsize)), label));
	
		if (driftThreshold != -1) {
			
			if (actDist > driftThreshold) {
				driftThreshold = -1;
				return true;
			}
			return false;
		}

		int samplingTime = sTimes;
		double[] samplingDist = new double[samplingTime];
		for (int i = 0; i < samplingTime; i++) {
			Collections.shuffle(shuffleIdx);
			samplingDist[i] = Math.log(getNNPSjd(nnps, new ArrayList<Integer>(shuffleIdx.subList(0, hBsize)), label));
		}

		double sampleMean = getMean(samplingDist);
		double sampleStd = getStd(samplingDist, sampleMean);
		if (sampleStd == 0) {
			driftThreshold = -1;
			return true;
		}
		NormalDistribution nd = new NormalDistribution(sampleMean, sampleStd);
		driftThreshold = nd.inverseCumulativeProbability(1 - ASL);
		if (actDist > driftThreshold) {
			driftThreshold = -1;
			return true;
		}
		return false;
	}

	public static double getNNPSjd(double[][] _nnps, ArrayList<Integer> _vec1Idx, ArrayList<Double> _label) {

		double minSum = 0, maxSum = 0;

		double totalL0_L0, totalL1_L1, tempV1L0_L0, tempV1L1_L1, tempV2L0_L0, tempV2L1_L1;
		for (int i = 0; i < _nnps.length; i++) {

			totalL0_L0 = 0;
			totalL1_L1 = 0;
			tempV1L0_L0 = 0;
			tempV1L1_L1 = 0;
			tempV2L0_L0 = 0;
			tempV2L1_L1 = 0;

			for (int idx : _vec1Idx) {
				if (_label.get(i) == 0 && _label.get(idx) == 0) {
					tempV1L0_L0 += _nnps[i][idx];
				} else if (_label.get(i) == 1 && _label.get(idx) == 1) {
					tempV1L1_L1 += _nnps[i][idx];
				}
			}

			for (int idx = 0; idx < _nnps.length; idx++) {
				if (_label.get(i) == 0 && _label.get(idx) == 0) {
					totalL0_L0 += _nnps[i][idx];
				} else if (_label.get(i) == 1 && _label.get(idx) == 1) {
					totalL1_L1 += _nnps[i][idx];
				}
			}

			tempV2L0_L0 = totalL0_L0 - tempV1L0_L0;
			tempV2L1_L1 = totalL1_L1 - tempV1L1_L1;

			if (_label.get(i) == 0) {
				if (tempV1L0_L0 > tempV2L0_L0) {
					minSum += tempV2L0_L0;
					maxSum += tempV1L0_L0;
				} else {
					minSum += tempV1L0_L0;
					maxSum += tempV2L0_L0;
				}

			} else {
				if (tempV1L1_L1 > tempV2L1_L1) {
					minSum += tempV2L1_L1;
					maxSum += tempV1L1_L1;
				} else {
					minSum += tempV1L1_L1;
					maxSum += tempV2L1_L1;
				}
			}
		}

		return 1 - minSum / maxSum;
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

	protected static double[][] buildNNPS_Weight(Instances _insts1, Instances _insts2, int _knn) throws Exception {

		ArrayList<Double> weight = new ArrayList<Double>();
		for (weka.core.Instance _i : _insts1) {
			weight.add(_i.weight());
		}
		for (weka.core.Instance _i : _insts2) {
			weight.add(_i.weight());
		}
		AddID instAddidFilter = new AddID();

		Instances merged = new Instances(_insts1);
		merged.addAll(_insts2);

		instAddidFilter.setInputFormat(merged);
		merged = Filter.useFilter(merged, instAddidFilter);
		merged.setClassIndex(0);
		merged.deleteAttributeAt(merged.numAttributes() - 1);

		KDTree instTree = new KDTree();
		instTree.getDistanceFunction().setOptions(new String[] { "-D" });
		instTree.setInstances(merged);

		ArrayList<Integer> tempRelatedSet;
		ArrayList<ArrayList<Integer>> relatedSets = new ArrayList<ArrayList<Integer>>();
		double[] instCount = new double[merged.numInstances()];

		Instances knnInsts;
		for (int i = 0; i < merged.numInstances(); i++) {

			knnInsts = instTree.kNearestNeighbours(merged.instance(i), _knn);

			tempRelatedSet = new ArrayList<Integer>();
			for (int j = 0; j < knnInsts.numInstances(); j++) {
				tempRelatedSet.add((int) knnInsts.instance(j).classValue() - 1);
			}
			tempRelatedSet.add((int) (merged.instance(i).classValue() - 1));

			relatedSets.add(tempRelatedSet);
			for (int id : tempRelatedSet) {
				instCount[id]++;
			}
		}

		double[][] instWeightMatrix = new double[merged.numInstances()][relatedSets.size()];

		for (int i = 0; i < relatedSets.size(); i++) {
			for (int id : relatedSets.get(i)) {
				// instWeightMatrix[id][i] = 1.0;
				instWeightMatrix[id][i] = weight.get(id) / instCount[id];
			}
		}

		return instWeightMatrix;
	}

	public static double getMean(double[] _data) {
		double mean = 0;
		for (double _d : _data) {
			mean += _d;
		}
		return mean / _data.length;
	}

	public static double getStd(double[] _data, double _mean) {
		double std = 0;
		for (double _d : _data) {
			std += (_d - _mean) * (_d - _mean);
		}
		std = std / (_data.length - 1);
		return Math.sqrt(std);
	}

}
