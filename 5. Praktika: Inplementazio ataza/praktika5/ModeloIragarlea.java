package praktika5;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class ModeloIragarlea {
	public static void main(String[] args) throws Exception{
		// 1. Argumentuen egiaztapena
		if (args.length!=4) {
			System.err.println("Erabilera: java -jar ModeloIragarlea.jar <NB.model> <train.arff> <test.arff> <iragarpenak.txt>");
			return;
		}
		String modelPath = args[0];
		String trainPath = args[1];
		String testPath = args[2];
		String iregarpenakPath = args[3];
		
		try {
			// 2. Datuak eta eredua kargatu
			Classifier model = (Classifier) SerializationHelper.read(modelPath);
			DataSource sourceTrain = new DataSource(trainPath);
			Instances trainData = sourceTrain.getStructure();
			if (trainData.classIndex()==-1) trainData.setClassIndex(trainData.numAttributes()-1);
			
			DataSource sourceTest = new DataSource(testPath);
			Instances testData = sourceTest.getDataSet();
			if (testData.classIndex()==-1) testData.setClassIndex(testData.numAttributes()-1);
			
			// 3. Egiaztatu egiturak
			if (!trainData.equalHeaders(testData)) {
				// 4. Test multzoa egokitu
				ArrayList<Integer> indexToDelete = new ArrayList<>();
				for (int i = 0; i<testData.numAttributes();i++) {
					Attribute a = trainData.attribute(testData.attribute(i).name());
					if (a == null && i != testData.classIndex()) indexToDelete.add(i);
				}
				int[] indexArray = indexToDelete.stream().mapToInt(i -> i).toArray();
				
				// Remove filtroa erabili ezgainbegiratua - unsupervised
				Remove r = new Remove();
				r.setAttributeIndicesArray(indexArray);
				r.setInvertSelection(false);
				r.setInputFormat(testData);
				
				testData = Filter.useFilter(testData, r);
			}
			
			// Azken konprobaketa
			if (!trainData.equalHeaders(testData)) {
				System.err.println("ERROREA: Ezin izan da Test multzoa Train egiturara egokitu.");
                System.err.println("Ziurtatu atributuen izenak berdinak direla bi fitxategietan.");
                return;
			}
			
			// 5. Iragarpenak egin
			PrintWriter pw = new PrintWriter(new FileWriter(iregarpenakPath));
			pw.println("Actual, Predicted, Distribution");
			
			for (int i=0; i < testData.numInstances();i++) {
				// Iragarpena
				double pred = model.classifyInstance(testData.instance(i));
				// Iragarpenaren izena
				String predString = testData.classAttribute().value((int) pred);
				// Probabilitate banaketa
				double[] dist = model.distributionForInstance(testData.instance(i));
				//'Actual' ? da (Test Blind delako), baina atributuan balioa badago idatzi dezakegu
				String actual = testData.instance(i).toString(testData.classIndex());
				
				pw.println(actual + ", " + predString + ", " + dist[(int) pred]);
				
			}
			
			pw.close();
			
			
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
}
