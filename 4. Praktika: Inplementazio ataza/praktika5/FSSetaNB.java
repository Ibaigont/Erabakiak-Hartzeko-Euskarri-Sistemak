package praktika5;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class FSSetaNB {
	public static void main(String[] args) throws Exception{
		// 1. Argumentuen egiaztapena
		if (args.length!=2) {
			System.err.println("Erabilera: java -jar FSSetaNB.jar <train.arff> <NB.model>");
			return;
		}
		String inPath = args[0];
		String modelPath = args[1];
		
		try {
			// 2. Datuak kargatu
			DataSource source = new DataSource(inPath);
			Instances data = source.getDataSet();
			
			// Klasea ezarri
			if (data.classIndex()==-1) data.setClassIndex(data.numAttributes()-1);
			
			// 3. Atributuen hautapena FSS - Feature Subset Selection
			AttributeSelection filtroa = new AttributeSelection();
			
			// Ebaluatzailea: CfsSubsetEval (Correlation-based Feature Selection)
			CfsSubsetEval eval = new CfsSubsetEval();
			
			// Bilaketa metodoa BestFirst
			BestFirst bF = new BestFirst();
			
			filtroa.setEvaluator(eval);
			filtroa.setSearch(bF);
			filtroa.setInputFormat(data);
			
			// Datuei filtroa aplikatu
			Instances dataFiltratuta = Filter.useFilter(data, filtroa);
			
			// 4. Eredua entrenatu (NaiveBayes)
			NaiveBayes nB = new NaiveBayes();			
			nB.buildClassifier(dataFiltratuta);
			
			// 5. Eredua gorde
			SerializationHelper.write(modelPath, nB);
			
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
}
