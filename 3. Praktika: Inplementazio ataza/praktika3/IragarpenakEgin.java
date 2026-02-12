package praktika3;

import java.io.FileWriter;
import java.io.PrintWriter;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class IragarpenakEgin {
    public static void main(String[] args) { // throws Exception kendu dugu try-catch erabiltzen dugulako
        // 1. Argumentuak egiaztatu
        if (args.length != 3) {
            System.err.println("Erabilera: java -jar IragarpenakEgin.jar <NB.model> <test_blind.arff> <test_predictions.txt>");
            return;
        }
        String modelPath = args[0];
        String testPath = args[1];
        String outputPath = args[2];

        PrintWriter writer = null; // Kanpoan deklaratu

        try {
            // 2. Eredua kargatu
            System.out.println("Eredua kargatzen...");
            Classifier cls = (Classifier) SerializationHelper.read(modelPath);

            // 3. Datuak kargatu
            System.out.println("Test datuak kargatzen...");
            DataSource source = new DataSource(testPath);
            Instances data = source.getDataSet();

            // 4. Klasea ezarri
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);

            // 5. Iragarpenak egin
            writer = new PrintWriter(new FileWriter(outputPath));
            writer.println("Instantzia_Zenbakia, Iragarritako_Klasea");
            
            System.out.println("Iragarpenak egiten...");
            for (int i = 0; i < data.numInstances(); i++) {
                double predIndex = cls.classifyInstance(data.instance(i));
                String predClassName = data.classAttribute().value((int) predIndex);
                writer.println((i + 1) + ", " + predClassName);
            }
            
            System.out.println("Eginda! Emaitzak hemen: " + outputPath);

        } catch (Exception e) {
            System.err.println("ERROREA GERTATU DA:");
            e.printStackTrace(); // Honek esango dizu zergatik huts egiten duen
        } finally {
            // 6. Fitxategia BETI itxi, errorea egon ala ez
            if (writer != null) {
                writer.close();
            }
        }
    }
}