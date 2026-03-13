package praktika6;

import java.io.File;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class SailkatzaileaSortu {
	public static void main(String[] args) {
		// 1. Argumentuaren egiaztapena
		if (args.length != 3) {
			System.err.println("Erabilera: java -jar SailkatzaileaSortu.jar <train_raw.arff> <train_BoW.arff> <eredua.model>");
			return;
		}
		String inPath = args[0];
		String outArffPath = args[1];
		String outModelPath = args[2];
		
		try {
			// 2. Datuak kargatu
			DataSource fitxategi = new DataSource(inPath);
			Instances data = fitxategi.getDataSet();
			
			// Klasea ezarri
			if (data.classIndex()==-1)
				data.setClassIndex(data.numAttributes()-1);
			
			// 3. StringToWordVector filtroa prestatu eta aplikatu
			// bow bag of words, Tf-idf Term frequency – Inverse document frequency)
			StringToWordVector filter = new StringToWordVector();
			
			// Filter definitu
			filter.setTFTransform(true);
			filter.setIDFTransform(true);
			filter.setLowerCaseTokens(true);
			filter.setWordsToKeep(1000);
			
			// Filter datuekin hasieratu
			filter.setInputFormat(data);	
			
			// Datuak filtratu
			// Train multzoak espazioaren transformazioa lortu
			Instances dataBoW = Filter.useFilter(data, filter);
			// Busca por nombre, si no lo encuentra usa el índice 0
			weka.core.Attribute classAttr = dataBoW.attribute(data.classAttribute().name());
			dataBoW.setClassIndex(classAttr != null ? classAttr.index() : 0);
			// 4. Datuak gorde
			ArffSaver saver = new ArffSaver();
			saver.setInstances(dataBoW);
			saver.setFile(new File(outArffPath));
			saver.writeBatch();
			
			// 5. Eredua entrenatu Naive bayes-ekin eraldatutako datuen gainean
			NaiveBayes nB = new NaiveBayes();
			nB.buildClassifier(dataBoW);
			
			// 6. Entrenatutako eredua gorde
			SerializationHelper.write(outModelPath, nB);
			
		}catch (Exception e) {
			e.printStackTrace();	
		}
	}
}
