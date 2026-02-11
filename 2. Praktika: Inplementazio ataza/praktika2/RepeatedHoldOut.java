package praktika2;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RepeatedHoldOut {
	public static void main(String[] args) throws Exception {
		// 1. argumentuaren egiaztapena
		if (args.length<2) {
			System.err.println("Erabilera: java -jar RepeatedHoldOut.jar <data.arff> <eval.txt>");
			return;
		}
		String inputPath = args[0];
		String outputPath = args[1];
		int n = 50; // Errepikapen kopurua
		
		try {
			// 2. Datuak kargatu
			DataSource fitxategi = new DataSource(inputPath);
			Instances data = fitxategi.getDataSet();
			// Klasea ezarri
			if (data.classIndex()==-1) {
				data.setClassIndex(data.numAttributes()-1);
			}
			// 3. Klase Minoritarioa ezarri
			int minClassIndex = -1;
            double minCount = Double.MAX_VALUE;
            int[] classCount = data.attributeStats(data.classIndex()).nominalCounts;

            for (int i = 0; i < classCount.length; i++) {
                if (classCount[i] < minCount) {
                    minCount = classCount[i];
                    minClassIndex = i;
                }
            }
            String minClassName = data.classAttribute().value(minClassIndex);
            
			//3. RepeatedHoldOut prestatu: Randomizatu eta zatitu (%66-an)
			double [] recalls = new double[n];
			for (int i=0;i<n;i++) {
				Instances dataKopia = new Instances(data);
				dataKopia.randomize(new Random(i));
				
				int trainTamaina = (int) (Math.round(data.numInstances()*0.66));
				int testTamaina = data.numInstances()-trainTamaina;
				
				Instances train = new Instances(dataKopia, 0, trainTamaina);
				Instances test = new Instances(dataKopia, trainTamaina, testTamaina);
				
				// 4. Eredua entrenatu Naive Bayes-ekin
				NaiveBayes nB = new NaiveBayes();
				nB.buildClassifier(train);
				
				// 5. Eredua burutu Test-arekin
				Evaluation eval = new Evaluation(train);
				eval.evaluateModel(nB, test);
				
				recalls[i]=eval.recall(minClassIndex);
			}		
			
			// 6. Estatistikak kalkulatu, batazbestekoa eta desbiderapena
			double sum = 0.0;
			for (double r : recalls) {
				sum +=r;
			}
			double batazbestekoa = sum / n;
			
			double desbiderapena = 0.0;
			for (double r : recalls) {
				desbiderapena += Math.pow(r- batazbestekoa, 2);
			}
			// Sample Standard Deviation formula: sqrt( sum((x - mean)^2) / (n - 1) )
            double stdev = Math.sqrt(desbiderapena / (n - 1));
			
			// 7. Emaitzak fitxategian idatzi
			
			PrintWriter pw = new PrintWriter(new FileWriter(outputPath));
            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");

            // a) Data eta Argumentuak
            pw.println("Exekuzio data: " + dtf.format(LocalDateTime.now()));
            pw.println("Argumentuak: Input=" + inputPath + ", Output=" + outputPath);
            pw.println("--------------------------------------------------");
            
            pw.println("Klase Minoriotarioa: " + minClassName);
            pw.println();
                       
            pw.printf("Recall (batazbestekoa): %.4f%n", batazbestekoa);
            pw.printf("F-score: %.4f%n", stdev);
            
            pw.println();
            
            pw.println("Iterazio guztien xehetasuna: ");
            for(int i=0; i<n;i++) pw.printf("%d: %.4f%n ", (i+1),recalls[i]);
            
            pw.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
}
