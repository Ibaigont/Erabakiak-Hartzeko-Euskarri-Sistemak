package praktika5;

import java.io.FileWriter;
import java.io.PrintWriter;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MetaSailkatzaile {
	public static void main(String[] args) throws Exception{
		// 1. Argumentuen egiaztapena
		if (args.length != 3) {
			System.err.println("Erabilera: java -jar MetaSailkatzaile.jar <train.arff> <test.arff> <predictions.txt>");
            return;
		}
		String trainPath = args[0];
        String testPath = args[1];
        String outputPath = args[2];
        
        try {
        	// 2. Datuak kargatu
        	DataSource sourceTrain = new DataSource(trainPath);
            Instances trainData = sourceTrain.getDataSet();
            if (trainData.classIndex() == -1) trainData.setClassIndex(trainData.numAttributes() - 1);

            DataSource sourceTest = new DataSource(testPath);
            Instances testData = sourceTest.getDataSet();
            if (testData.classIndex() == -1) testData.setClassIndex(testData.numAttributes() - 1);
            
            // 3. Meta Sailkatzailea konfiguratu
            AttributeSelectedClassifier cls = new AttributeSelectedClassifier();
            
            // Ebaluatzailea
            CfsSubsetEval eval = new CfsSubsetEval();
            
            // Bilaketa metodoa
            BestFirst search = new BestFirst();
            
            // Sailkatzailea
            NaiveBayes nB = new NaiveBayes();
            
            // Dena meta sailkatzaileari esletu
            cls.setEvaluator(eval);
            cls.setSearch(search);
            cls.setClassifier(nB);
            
            // 4. Entrenamendua
            
            cls.buildClassifier(trainData);
            
            // 5. Iragarpenak
            PrintWriter pw = new PrintWriter(new FileWriter(outputPath));
            pw.println("Actual, Predicted");
            
            for (int i = 0; i < testData.numInstances(); i++) {
            	double pred = cls.classifyInstance(testData.instance(i));
            	String predString = trainData.classAttribute().value((int) pred);
            	String actual = "?";
            	try {
            		if(!testData.instance(i).classIsMissing()) {
            			actual = testData.instance(i).stringValue(testData.classIndex());
            		}
            	}catch (Exception e) {
            		actual = "?";
				}
            	pw.println(actual + ", " + predString);
            }
            
            pw.close();
            
        }catch (Exception e) {
			e.printStackTrace();
		}
	}
}
