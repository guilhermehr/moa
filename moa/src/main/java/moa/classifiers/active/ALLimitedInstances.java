/*
 *    ALRandom.java
 *    Copyright (C) 2016 Otto von Guericke University, Magdeburg, Germany
 *    @author Daniel Kottke (daniel dot kottke at ovgu dot de)
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
package moa.classifiers.active;

import java.util.LinkedList;
import java.util.List;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.active.budget.BudgetManager;
import moa.core.Measurement;
import moa.options.ClassOption;

public class ALLimitedInstances extends AbstractClassifier implements ALClassifier {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Active learning classifier for evolving data streams based on limited number of instances";
    }
    
    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "drift.SingleClassifierDrift");

    public ClassOption budgetManagerOption = new ClassOption("budgetManager",
            'b', "BudgetManager that should be used.",
            BudgetManager.class, "FixedBM");

    /**
     * Option for limit the number of botnet instances checked
     */
    public IntOption numLimitedInstancesOption = new IntOption("numLimitedInstances",
            'y', "Number of limited instances presented to the classifier. Zero (0) no limit.",
            0, 0, Integer.MAX_VALUE);
    
    public Classifier classifier;

    public BudgetManager budgetManager;
    
    /**
     * Current number of botnet instances checked
     */
    private int botnetInstances;
    
    /**
     * Block the train on instances based on the numLimitedInstances for botnet
     */
    private boolean blockTrain = false;
    
	@Override
	public int getLastLabelAcqReport() {
		return budgetManager.getLastLabelAcqReport();
	}
	
	@Override
	public boolean isRandomizable() {
        return true;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		return this.classifier.getVotesForInstance(inst);
	}

	@Override
	public void resetLearningImpl() {
        this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
        this.classifier.resetLearning();
        this.budgetManager = ((BudgetManager) getPreparedClassOption(this.budgetManagerOption));
        this.budgetManager.resetLearning();
        
        /**
         * Reset number of current botnet instances checked to zero
         */
        this.botnetInstances = 0;
        
        /**
         * Reset the block of train on instances based on the numLimitedInstances for botnet
         */
        this.blockTrain = false;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		
        //Whether botnet instance and has numLimitedInstances
		if (!blockTrain && this.numLimitedInstancesOption.getValue() != 0 && inst.classValue() == 1.0) {
			
			//not reached number of limited instances do nothing
			if (this.botnetInstances < this.numLimitedInstancesOption.getValue()) {
				this.botnetInstances++;
				System.out.println("botnet: " + this.botnetInstances);
			} else {
				blockTrain = true;
			}
		}
		
		if (blockTrain) {
			return;
		}
		
        this.classifier.trainOnInstance(inst);
		
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
        List<Measurement> measurementList = new LinkedList<Measurement>();
        return measurementList.toArray(new Measurement[measurementList.size()]);
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
        ((AbstractClassifier) this.classifier).getModelDescription(out, indent);
		
	}
	
	@Override
	public void setModelContext(InstancesHeader ih) {
		super.setModelContext(ih);
		classifier.setModelContext(ih);
	}
}
