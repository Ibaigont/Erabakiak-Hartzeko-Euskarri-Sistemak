package praktika5;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.ReplaceWithMissingValue;

public class Stratified70percentSplit {
	public static void main(String[] args) throws Exception{
		// 1. Argumentuen egiaztapena
		if (args.length!=3) {
			System.err.println("Erabilera: java -jar Stratified70percentSplit.jar <input.arff> <output_train.arff> <output_test_blind.arff>");
			return;
		}
		String inPath = args[0];
		String outTrainPath = args[1];
		String outTestPath = args[2];
		
		try {
			// 2. Datuak kargatu
			DataSource source = new DataSource(inPath);
			Instances data = source.getDataSet();
			
			// Klasea ezarri
			if (data.classIndex()==-1) data.setClassIndex(data.numAttributes()-1);
			
			// 3. Estratifikazioa eta zatiketa %70an
			// Resample filtro gainbegiratua (supervised)
			Resample resample = new Resample();
			resample.setInputFormat(data);
			resample.setSampleSizePercent(70.0);
			resample.setNoReplacement(true);
			resample.setBiasToUniformClass(0.0);
			resample.setRandomSeed(1);
			
			// Train multzoa sortu
			resample.setInvertSelection(false);
			Instances trainData = Filter.useFilter(data, resample);
			
			// Test multzoa sortu
			resample.setInvertSelection(true);
			Instances testData = Filter.useFilter(data, resample);
			
			// 4. TEST MULTZOA "ITSU" BIHURTU (Blind Test)
			ReplaceWithMissingValue missingFilter = new ReplaceWithMissingValue();
			// Filtro honek indizeak 1etik hasita (1-based index) erabiltzen ditu String formatuan.
            // Klasearen indizea lortu eta +1 egin behar dugu Weka-ren filtroari pasatzeko.
            String classIndexStr = "" + (testData.classIndex() + 1);
            
            missingFilter.setAttributeIndices(classIndexStr);
            missingFilter.setInputFormat(testData);
            
            Instances testBlindData = Filter.useFilter(testData, missingFilter);
            
            // 5. FITXATEGIAK GORDE
            // Train
            DataSink.write(outTrainPath, trainData);
            System.out.println("Train gorde da: " + outTrainPath);
            
            // Test Blind
            DataSink.write(outTestPath, testBlindData);
            System.out.println("Test Blind gorde da: " + outTestPath);
			
			
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
}
