package sample;

import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.input.MouseEvent;
import javafx.scene.media.*;
import javafx.util.Duration;
import org.apache.commons.io.FileUtils;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.codec.reader.CodecRecordReader;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;


public class MainWindowController {
    private static final String VID_FOLDER = "DL4JVideoShapesExample/";
    private static final String VID_URL = "/DL4JVideoShapesExample/shapes_";
    private static final String NET_CONF_PATH = "DL4JVideoShapesExample/videoconf.json";
    private static final String NET_PARAM_PATH = "DL4JVideoShapesExample/videomodel.bin";

    @FXML public MediaView mediaView;
    @FXML public Label prediction;
    @FXML public ChoiceBox<String> videoNumber;

    private MultiLayerNetwork net;
    private INDArray predicted;
    private HashMap<Integer, String> labelMap = new HashMap<Integer, String>() {
        {
            put(0, "circle");
            put(1, "square");
            put(2, "arc");
            put(3, "line");
        }
    };

    public MainWindowController() {
        try (final DataInputStream dis = new DataInputStream(new FileInputStream(NET_PARAM_PATH))) {
            final INDArray params = Nd4j.read(dis);
            final MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File(NET_CONF_PATH)));

            net = new MultiLayerNetwork(conf, params);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @FXML
    public void initialize() {
        final ArrayList<String> videos = new ArrayList<>();
        for (int i = 0; i < 500; i++) {
            videos.add(Integer.toString(i));
        }

        videoNumber.setItems(FXCollections.observableArrayList(videos));
    }

    public void loadVideo(MouseEvent mouseEvent) {
        try {
            final String fullUrl = "file://" + Paths.get("").toAbsolutePath().toString() + VID_URL + videoNumber.getValue() + ".mp4";
            final MediaPlayer player = new MediaPlayer(new Media(fullUrl));

            prediction.setText("Analyzing...");

            final DataSetIterator dataSetIterator = getDataSetIterator(VID_FOLDER, Integer.parseInt(videoNumber.getValue()), 1, 1);
            final DataSet dsTest = dataSetIterator.next();
            predicted = net.output(dsTest.getFeatureMatrix(), false);

            mediaView.setMediaPlayer(player);


            final Timeline tl = new Timeline(
                    new KeyFrame(Duration.ZERO, (ae) -> showFrame()),
                    new KeyFrame(Duration.millis(25))
            );
            tl.setCycleCount(Timeline.INDEFINITE);

            player.setOnEndOfMedia(tl::stop);

            player.play();
            tl.play();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private int maxIndex(double[] array) {
        int max = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > array[max]) {
                max = i;
            }
        }
        return max;
    }

    private void showFrame() {
        final int currentPos = (int)(mediaView.getMediaPlayer().getCurrentTime().toSeconds() * 25.0);

        if (currentPos < 150) {
            final double predCircle = predicted.getRow(0).getDouble(0, currentPos);
            final double predSquare = predicted.getRow(0).getDouble(1, currentPos);
            final double predArc = predicted.getRow(0).getDouble(2, currentPos);
            final double predLine = predicted.getRow(0).getDouble(3, currentPos);

            final double[] preds = new double[]{predCircle, predSquare, predArc, predLine};

            prediction.setText(labelMap.get(maxIndex(preds)));
        }
    }

    private static DataSetIterator getDataSetIterator(String dataDirectory, int startIdx, int nExamples, int miniBatchSize) throws Exception {
        final SequenceRecordReader featuresTrain = getFeaturesReader(dataDirectory, startIdx, nExamples);
        final SequenceRecordReader labelsTrain = getLabelsReader(dataDirectory, startIdx, nExamples);

        final SequenceRecordReaderDataSetIterator sequenceIter = new SequenceRecordReaderDataSetIterator(featuresTrain, labelsTrain, miniBatchSize, 4, false);
        sequenceIter.setPreProcessor(new VideoPreProcessor());

        return sequenceIter;
    }

    private static SequenceRecordReader getFeaturesReader(String path, int startIdx, int num) throws Exception {
        final InputSplit is = new NumberedFileInputSplit(path + "shapes_%d.mp4", startIdx, startIdx + num - 1);

        final Configuration conf = new Configuration();
        conf.set(CodecRecordReader.RAVEL, "true");
        conf.set(CodecRecordReader.START_FRAME, "0");
        conf.set(CodecRecordReader.TOTAL_FRAMES, String.valueOf(150));
        conf.set(CodecRecordReader.ROWS, String.valueOf(130));
        conf.set(CodecRecordReader.COLUMNS, String.valueOf(130));

        final CodecRecordReader crr = new CodecRecordReader();
        crr.initialize(conf, is);

        return crr;
    }

    private static SequenceRecordReader getLabelsReader(String path, int startIdx, int num) throws Exception {
        final InputSplit isLabels = new NumberedFileInputSplit(path + "shapes_%d.txt", startIdx, startIdx + num - 1);
        final CSVSequenceRecordReader csvSeq = new CSVSequenceRecordReader();
        csvSeq.initialize(isLabels);

        return csvSeq;
    }

    private static class VideoPreProcessor implements DataSetPreProcessor {
        @Override
        public void preProcess(DataSet toPreProcess) {
            toPreProcess.getFeatureMatrix().divi(255);  //[0,255] -> [0,1] for input pixel values
        }
    }
}