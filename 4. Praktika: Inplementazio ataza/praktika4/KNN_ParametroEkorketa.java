package praktika4;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;

public class KNN_ParametroEkorketa {
	public static void main(String[] args) throws Exception{
		// 1. argumentuen egiaztapena
		if (args.length<2){
			System.err.println("Erabilera: java -jar KNN_ParametroEkorketa.jar <data.arff> <emaitzak.txt>");
			return;
		}
		String inPath = args[0];
		String outPath = args[1];
		
		try {
			// 2. Datuak kargatu
			DataSource source = new DataSource(inPath);
			Instances data = source.getDataSet();
			
			// Klasea ezarri
			if (data.classIndex()==-1) {
				data.setClassIndex(data.numAttributes()-1);
			}
			
			//3. HoldOut prestatu: Randomizatu eta zatitu %66-an
			data.randomize(new Random(1));
			int trainTamaina = (int)(Math.round(data.numInstances()*0.66));
			int testTamaina = data.numInstances()-trainTamaina;
			
			Instances train = new Instances(data, 0, trainTamaina);
			Instances test = new Instances(data, trainTamaina, testTamaina);
			
			//4. Parametro ekorketa (k, w, d)
			// Aldagaiak emaitza onenak gordetzeko
			double bestMetric = -1.0;
			Evaluation bestEval = null;
			
			// Parametro optimoak gordetzeko
			int bestK = 0;
			String bestD = "";
			String bestW = "";
			
			// k begizta 1etik 20ra
			for (int k=1; k<=20; k++) {
				// w begizta (Ponderazioa): None, Inverse, Similarity
				int[] wAukerak = {IBk.WEIGHT_NONE, IBk.WEIGHT_INVERSE, IBk.WEIGHT_SIMILARITY};
				
				for (int wVal : wAukerak) {
					// d begizta (Metrika): Euclidean, Manhattan
					DistanceFunction[] dAukerak = {new EuclideanDistance(), new ManhattanDistance()};
					for (DistanceFunction dVal : dAukerak) {
						IBk kNN = new IBk();
						kNN.setKNN(k);
						kNN.setDistanceWeighting(new SelectedTag(wVal, IBk.TAGS_WEIGHTING));
						kNN.getNearestNeighbourSearchAlgorithm().setDistanceFunction(dVal);
						
						// Entrenatu
						kNN.buildClassifier(train);
						
						// Ebaluatu
						Evaluation eval = new Evaluation(train);
						eval.evaluateModel(kNN, test);
						
						// Optimizazio irizpidea: Weighted Average F-Score
						double unekoMetric = eval.weightedFMeasure();
						if (unekoMetric > bestMetric) {
							bestMetric = unekoMetric;
							bestEval = eval;
							bestK = k;
							bestW = new SelectedTag(wVal, IBk.TAGS_WEIGHTING).getSelectedTag().getReadable();
						}
					}
				}
			}
			// 5. Klase minoritarioa identifikatu datu originaletan
			int minClassIndex = -1;
			double minCount = Double.MAX_VALUE;
			int[] classCount = data.attributeStats(data.classIndex()).nominalCounts;
			
			for (int i=0; i< classCount.length; i++) {
				if (classCount[i]< minCount) {
					minCount = classCount[i];
					minClassIndex = i;
				}
			}
			String minClassName = data.classAttribute().value(minClassIndex);
			// 6. Emaitzak fitxategian idatzi
			PrintWriter pw = new PrintWriter(new FileWriter(outPath));
			DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
			
			// Data eta argumentuak
			pw.println("Exekuzio data: " + dtf.format(LocalDateTime.now()));
			pw.println("Argumentuak: Input=" + inPath + ", Output=" + outPath);
			pw.println("--------------------------------------------------");
			// Lortu ditugun parametro optimoak
			pw.println("PARAMETRO OPTIMOAK: ");
			pw.println("k (Auzokideak): "+ bestK);
			pw.println("w (Ponderazioa): " + bestW);
			pw.println("d (Metrika): " + bestD);
			pw.println("--------------------------------------------------");
			
			// Klase minoritarioaren metrikak
			pw.println("KLase Minoritaria: "+ minClassName);
			pw.printf("Precision: %.4f%n", bestEval.precision(minClassIndex));
			pw.printf("Recall: %.4f%n", bestEval.recall(minClassIndex));
			pw.printf("F-Score: %.4f%n", bestEval.fMeasure(minClassIndex));
			
			// Weighted Average Metrikak
			pw.println("Weighted Average Metrikak: ");
			pw.printf("Precision: %.4f%n", bestEval.weightedPrecision());
			pw.printf("Recall: %.4f%n", bestEval.weightedRecall());
			pw.printf("F-Score: %.4f%n", bestEval.weightedFMeasure());
			
			// Nahasmen Matrizea
			pw.println("Nahasmen Matrizea: ");
			pw.println(bestEval.toMatrixString());
			
			pw.close();
			
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
}
