/*
 *    ALPrequentialEvaluationTask.java
 *    Copyright (C) 2017 Otto-von-Guericke-University, Magdeburg, Germany
 *    @author Cornelius Styp von Rekowski (cornelius.styp@ovgu.de)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.tasks.meta;

import java.awt.Color;
import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.classifiers.active.ALClassifier;
import moa.classifiers.active.ALLimitedInstances;
import moa.classifiers.meta.OzaBagAdwin;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.Example;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.TimingUtils;
import moa.evaluation.ALClassificationPerformanceEvaluator;
import moa.evaluation.ConfusionMatrix;
import moa.evaluation.LearningEvaluation;
import moa.evaluation.preview.LearningCurve;
import moa.evaluation.preview.PreviewCollectionLearningCurveWrapper;
import moa.options.ClassOption;
import moa.streams.ExampleStream;
import moa.tasks.TaskMonitor;

/**
 * This task performs prequential evaluation for an active learning classifier 
 * (testing, then training with each example in sequence). It is mainly based
 * on the class EvaluateALPrequentialCV.
 * 
 * @author Cornelius Styp von Rekowski (cornelius.styp@ovgu.de)
 * @version $Revision: 1 $
 */
public class ALPrequentialEvaluationTask extends ALMainTask {
	
	private static final long serialVersionUID = 1L;
	
	@Override
	public String getPurposeString() {
		return "Perform prequential evaluation (testing, then training with"
				+ " each example in sequence) for an active learning"
				+ " classifier.";
	}
	
	public ClassOption learnerOption = new ClassOption("learner", 'l',
            "Learner to train.", ALClassifier.class, 
            "moa.classifiers.active.ALRandom");
	
	public ClassOption streamOption = new ClassOption("stream", 's',
            "Stream to learn from.", ExampleStream.class,
            "generators.RandomTreeGenerator");
	
	public ClassOption evaluatorOption = new ClassOption(
			"evaluator", 'e',
            "Active Learning classification performance evaluation method.",
            ALClassificationPerformanceEvaluator.class,
            "ALWindowClassificationPerformanceEvaluator");
	
	public IntOption instanceLimitOption = new IntOption("instanceLimit", 'i',
            "Maximum number of instances to test/train on  (-1 = no limit).",
            100000000, -1, Integer.MAX_VALUE);

	public IntOption timeLimitOption = new IntOption("timeLimit", 't',
            "Maximum number of seconds to test/train for (-1 = no limit).", -1,
            -1, Integer.MAX_VALUE);
	
	public IntOption sampleFrequencyOption = new IntOption("sampleFrequency",
            'f',
            "How many instances between samples of the learning performance.",
            100000, 0, Integer.MAX_VALUE);
	
	public FileOption dumpFileOption = new FileOption("dumpFile", 'd',
            "File to append intermediate csv results to.", null, "csv", true);
	
	public FlagOption printTreeOption = new FlagOption("printTree", 'a', "Option to print tree.");
	

	public IntOption classNumberOption = new IntOption("classNumber", 'c',
            "Number of classes to test/train on  (2 = default).",
            2, 2, Integer.MAX_VALUE);	
	
	
	/**
	 * Constructor which sets the color coding to black.
	 */
	public ALPrequentialEvaluationTask() {
		this(Color.BLACK);
	}
	
	/**
	 * Constructor with which a color coding can be set.
	 * @param colorCoding the color used by the task
	 */
	public ALPrequentialEvaluationTask(Color colorCoding) {
		this.colorCoding = colorCoding;
	}
	
	@Override
	public Class<?> getTaskResultType() {
		return LearningCurve.class;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
		// get stream
		ExampleStream<Example<Instance>> stream = 
				(ExampleStream<Example<Instance>>) 
				getPreparedClassOption(this.streamOption);
		
		// initialize learner
		ALClassifier learner = 
				(ALClassifier) getPreparedClassOption(this.learnerOption);
		learner.resetLearning();
		learner.setModelContext(stream.getHeader());
		
		// get evaluator
        ALClassificationPerformanceEvaluator evaluator = (ALClassificationPerformanceEvaluator) 
        		getPreparedClassOption(this.evaluatorOption);
        
        // initialize learning curve
        LearningCurve learningCurve = new LearningCurve(
        		"learning evaluation instances");
        
        
        int n = classNumberOption.getValue();
        
        //Confusion Matrix
        ConfusionMatrix cm = new ConfusionMatrix();
        
//        cm.increaseValue("0", "0");
//        cm.increaseValue("0", "1");
//        cm.increaseValue("1", "1");
//        cm.increaseValue("1", "0");

        for(int i = 0; i < n; i++) {
        	for(int j = 0; j < n; j++) {
        		cm.increaseValue(String.valueOf(i), String.valueOf(j));
        	}
        }
        
        double[][] matrixTotal = new double[n][n];
        double[][] matrixTemp = new double[n][n];
                
        
//        double colunaATotal = 0;
//        double colunaBTotal = 0;
//        double linhaATotal = 0;
//        double linhaBTotal = 0;
//        
//        double linhaA = 0;
//        double linhaB = 0;
//        double colunaA = 0;
//        double colunaB = 0;
        
        
        boolean first = true;
        
		// perform training and testing
        int maxInstances = this.instanceLimitOption.getValue();
        int instancesProcessed = 0;
        int maxSeconds = this.timeLimitOption.getValue();
        int secondsElapsed = 0;
        boolean firstDump = true;
        boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        long lastEvaluateStartTime = evaluateStartTime;
        double RAMHours = 0.0;
        
        File dumpFile = this.dumpFileOption.getFile();
        PrintStream immediateResultStream = null;
        if (dumpFile != null) {
        	try {
        		if (dumpFile.exists()) {
        			immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile, true), true);
                } else {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile), true);
                }
        	} catch (Exception ex) {
                throw new RuntimeException(
                		"Unable to open immediate result file: " + dumpFile, ex);
            }
        }
        
        monitor.setCurrentActivity("Evaluating learner...", -1.0);
        while (stream.hasMoreInstances()
        	   && ((maxInstances < 0) 
        		   || (instancesProcessed < maxInstances))
        	   && ((maxSeconds < 0) || (secondsElapsed < maxSeconds))) 
        {
        	Example<Instance> trainInst = stream.nextInstance();
        	Example<Instance> testInst = trainInst;
        	
        	// predict class for instance
        	double[] prediction = learner.getVotesForInstance(testInst);
        	evaluator.addResult(testInst, prediction);
        	
        	if(instancesProcessed+2 == 1299552) {
        		System.out.println("");
        	}
        	
        	// train on instance
        	learner.trainOnInstance(trainInst);
        	
        	// check if label was acquired
        	int labelAcquired = learner.getLastLabelAcqReport();
        	evaluator.doLabelAcqReport(trainInst, labelAcquired);
        	
        	instancesProcessed++;
        	
            cm = evaluator.addResult(testInst, prediction, cm);
            
            for(int i = 0; i < n; i++) {
            	for(int j = 0; j < n; j++) {
            		matrixTemp[i][j] = cm.getRowCM(String.valueOf(i), String.valueOf(j)) - matrixTotal[i][j];
            	}
            }
                        
        	
        	// update learning curve
        	if (instancesProcessed % this.sampleFrequencyOption.getValue() == 0
        		|| !stream.hasMoreInstances())
        	{
        		long evaluateTime = 
        				TimingUtils.getNanoCPUTimeOfCurrentThread();
                double time = TimingUtils.nanoTimeToSeconds(
                		evaluateTime - evaluateStartTime);
                double timeIncrement = TimingUtils.nanoTimeToSeconds(
                		evaluateTime - lastEvaluateStartTime);
                double RAMHoursIncrement = 
                		learner.measureByteSize() / (1024.0 * 1024.0 * 1024.0); //GBs
                RAMHoursIncrement *= (timeIncrement / 3600.0); //Hours
                RAMHours += RAMHoursIncrement;
                lastEvaluateStartTime = evaluateTime;
        		
                
                List<Measurement> measurements = new ArrayList<Measurement>();
                
                measurements.add(new Measurement(
						"learning evaluation instances",
						instancesProcessed));
                
                measurements.add(new Measurement(
                        "evaluation time ("
                        + (preciseCPUTiming ? "cpu "
                        : "") + "seconds)",
                        time));
                
                measurements.add(new Measurement(
                        "model cost (RAM-Hours)",
                        RAMHours));
                
                measurements.add(new Measurement("Total Geral", cm.getTotalSum()));

                double tp = 0, tn = 0, fp = 0, fn = 0;
                int classIndex = 0;
                
                tp = (first ? matrixTemp[classIndex][classIndex] - 1 : matrixTemp[classIndex][classIndex]);
                
                for(int l = 0; l < n; l++) {
                	fp += (first ? matrixTemp[l][classIndex] - 1 : matrixTemp[l][classIndex]);
                	fn += (first ? matrixTemp[classIndex][l] - 1 : matrixTemp[classIndex][l]);
                }
                
                fp -= tp;
                fn -= tp;
                
        		measurements.add(new Measurement("TP", tp));
        		measurements.add(new Measurement("FP", fp));
        		measurements.add(new Measurement("FN", fn));
                
                for(int l = 0; l < n; l++) {
                	for(int k = 0; k < n; k++) {
                		tn += (first ? matrixTemp[l][k] - 1 : matrixTemp[l][k]);
                	}
                }
                
                tn = tn - tp -fp - fn;

        		measurements.add(new Measurement("TN", tn));
                
                
                for(int i = 0; i < n; i++) {
                	for(int j = 0; j < n; j++) {
                		measurements.add(new Measurement(String.format("%dx%d", i, j), (first ? matrixTemp[i][j] - 1 : matrixTemp[i][j])));
                		matrixTotal[i][j] = cm.getRowCM(String.valueOf(i), String.valueOf(j));
                		matrixTemp[i][j] = 0;
                	}
                }

        		learningCurve.insertEntry(new LearningEvaluation(measurements.toArray(new Measurement[measurements.size()]),
        				evaluator, learner));                
                
                first = false;
                        		
        		if (immediateResultStream != null) {
                    if (firstDump) {
                        immediateResultStream.println(learningCurve.headerToString());
                        firstDump = false;
                    }
                    immediateResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
                    immediateResultStream.flush();
                }
        	}
        	
        	// update monitor
        	if (instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0 && learningCurve.numEntries() > 0) {
        		if (monitor.taskShouldAbort()) {
                    return null;
                }
        		
        		long estimatedRemainingInstances = 
        				stream.estimatedRemainingInstances();
        		
        		if (maxInstances > 0) {
        			long maxRemaining = maxInstances - instancesProcessed;
        			if ((estimatedRemainingInstances < 0 || estimatedRemainingInstances == 0)
        				|| (maxRemaining < estimatedRemainingInstances))
        			{
        				estimatedRemainingInstances = maxRemaining;
        			}
        		}
        		
        		
        		// calculate completion fraction
        		double fractionComplete = (double) instancesProcessed / 
        				(instancesProcessed + estimatedRemainingInstances);
        		monitor.setCurrentActivityFractionComplete(
        				estimatedRemainingInstances < 0 ? 
        						-1.0 : fractionComplete);
        		
        		
        		// TODO currently the preview is sent after each instance
        		// 		should be changed later on
        		if (monitor.resultPreviewRequested() || isSubtask()) {
        			monitor.setLatestResultPreview(new PreviewCollectionLearningCurveWrapper((LearningCurve)learningCurve.copy(), this.getClass()));
                }
        		
        		// update time measurement
        		secondsElapsed = (int) TimingUtils.nanoTimeToSeconds(
        				TimingUtils.getNanoCPUTimeOfCurrentThread()
                        - evaluateStartTime);
        	}
        }
        
        //TODO
        if(this.printTreeOption.isSet()) {
            System.out.println("ALPrequentialEvaluationTask... ");
            //System.out.println(learner);
            
            Classifier classifier = ((ALLimitedInstances) learner).getClassifier();
            
            if(classifier instanceof OzaBagAdwin) {
                OzaBagAdwin oza = (OzaBagAdwin) (classifier);

                System.out.println("Ozabagadwin: ");
                System.out.println(oza);
                
                int i = 0;
                
                for(Classifier c : oza.getSubClassifiers()) {
                	
                	HoeffdingTree ht = (HoeffdingTree) c;
                	
                	System.out.println("Oza Hoeffding Tree " + i++ + ":");
                	System.out.println();
                	System.out.println(ht);
                	
                }
                
            } else if (classifier instanceof HoeffdingTree) {
            	
            	HoeffdingTree ht = (HoeffdingTree) (classifier);
            	
                System.out.println("HoeffdingTree: ");
                System.out.println(ht);
            }
        }    
        
        
        if (immediateResultStream != null) {
            immediateResultStream.close();
        }
		
		return new PreviewCollectionLearningCurveWrapper(learningCurve, this.getClass());
	}
	
	@Override
	public List<ALTaskThread> getSubtaskThreads() {
		return new ArrayList<ALTaskThread>();
	}
}
