package com.mbticlassifierapp;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Bundle;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.github.mikephil.charting.charts.BarChart;
import com.github.mikephil.charting.charts.HorizontalBarChart;
import com.github.mikephil.charting.components.Description;
import com.github.mikephil.charting.data.BarData;
import com.github.mikephil.charting.data.BarDataSet;
import com.github.mikephil.charting.data.BarEntry;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.LiteModuleLoader;

public class SpeechActivity extends AppCompatActivity {
    String TAG = "SpeechPage";

    private Boolean captureStatus = false;
    private String captureText;

    private String[] REQUIRED_PERMISSIONS = new String[]{
            Manifest.permission.RECORD_AUDIO,
    };
    private static final int REQUEST_CODE_PERMISSIONS = 1;

    private ImageView recordInactive;
    private ImageView recordActive;
    private TextView speechText;
    private Button captureButton;
    private TextView resultText;
    private TextView confidenceText;

    private HorizontalBarChart barChart;
    private final int numPlot = 5;
    private final int[] colorList = new int[]{Color.RED, Color.BLUE, Color.GREEN, Color.MAGENTA, Color.CYAN};

    private SpeechRecognizer speechRecognizer;

    private String modelName = "bertMBTIClassifier.ptl";
    private String vocabName = "vocab.txt";
    private Module module;
    private HashMap<String, Integer> mTokenIdMap = null;
    private HashMap<Integer, String> mIdTokenMap = null;

    private int tokenLimit = 510;
    private String CLS = "[CLS]";
    private String SEP = "[SEP]";
    private String PAD = "[PAD]";
    private final String[] MBTI = {"INTJ", "INTP", "INFJ", "INFP",
            "ISTJ", "ISTP", "ISFJ", "ISFP",
            "ENTJ", "ENTP", "ENFJ", "ENFP",
            "ESTJ", "ESTP", "ESFJ", "ESFP"};
    private final int numLabels = 16;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_speech);
        Log.i(TAG, "Successfully Created activity_speech");

        recordInactive = findViewById(R.id.recordInactive);
        recordActive = findViewById(R.id.recordActive);
        speechText = findViewById(R.id.speechText);

        captureButton = findViewById(R.id.captureButton);
        resultText = findViewById(R.id.Result_Text);
        confidenceText = findViewById(R.id.Confidence_Text);

        barChart = findViewById(R.id.barChart);
        barChart.setNoDataTextColor(Color.BLACK);

        captureButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                if (captureStatus == false){
                    captureStatus = true;
                    captureButton.setText("STOP");

                    recordInactive.setAlpha((float) 0);
                    recordActive.setAlpha((float) 1);

                    resultText.setText("Detection in Progress..");
                    confidenceText.setText("Confidence : 0");
                    startAudioCapture();
                } else{
                    captureStatus = false;
                    captureButton.setText("START");

                    recordInactive.setAlpha((float) 1);
                    recordActive.setAlpha((float) 0);

                    endAudioCapture();
                    Log.i(TAG, "Speech Text: "+speechText.getText().toString());
//                    classifyMBTI(speechText.getText().toString());
                    speechText.setText("");
                    captureText = "";
                }
            }
        });

        try{
            module = LiteModuleLoader.load(assetFilePath(this, modelName));
            Log.i(TAG, "module loading successful!");
        } catch(IOException e){
            Log.e(TAG, "module loading failed", e);
        }

        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(vocabName)));
            String line;
            mTokenIdMap = new HashMap<>();
            mIdTokenMap = new HashMap<>();
            int count = 0;
            while (true) {
                line = br.readLine();
                if (line != null) {
                    mTokenIdMap.put(line, count);
                    mIdTokenMap.put(count, line);
                    count++;
                } else {
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "vocab.txt load failed", e);
        }

    }

    @Override
    protected void onStart(){
        super.onStart();
        if(!checkPermissions()){
            requestPermission();
            Log.i(TAG, "Successfully entered onStart_Check");
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (speechRecognizer != null) {
            speechRecognizer.destroy();
        }
    }

    private boolean checkPermissions(){
        for(String permission: REQUIRED_PERMISSIONS){
            int permissionState = ActivityCompat.checkSelfPermission(this, permission);
            if (permissionState != PackageManager.PERMISSION_GRANTED) return false;
        }
        return true;
    }

    private void requestPermission(){
        ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults){
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        Log.i(TAG, "onRequestPermissionResult");
        if (requestCode == REQUEST_CODE_PERMISSIONS){
            if (grantResults.length <= 0){
                Log.i(TAG, "User interaction was cancelled.");
            }
            else if (grantResults[0] == PackageManager.PERMISSION_GRANTED){
                //Permission Granted.
                Log.i(TAG, "Permission Successfully Granted!");
            }
            else{
                //Permission Denied.
                Log.e(TAG, "Permission denied!");
            }
        }
    }

    private String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            Log.i(TAG, "Absol path: " + file.getAbsolutePath());
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                Log.i(TAG, "Buffer writing done!");
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    private void startAudioCapture(){
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US");
        intent.putExtra(RecognizerIntent.EXTRA_PROMPT, "Speak something...");

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(getApplicationContext());
        speechRecognizer.setRecognitionListener(new RecognitionListener() {
            @Override
            public void onReadyForSpeech(Bundle params) {

            }

            @Override
            public void onBeginningOfSpeech() {

            }

            @Override
            public void onRmsChanged(float rmsdB) {

            }

            @Override
            public void onBufferReceived(byte[] buffer) {
//                String str = new String(buffer);
//                String originText = speechText.getText().toString();
//                speechText.setText(originText+" " + str);
            }

            @Override
            public void onEndOfSpeech() {

            }

            @Override
            public void onError(int error){

            }

            @Override
            public void onResults(Bundle bundle) {
                ArrayList<String> captureResult = bundle.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                String originText = speechText.getText().toString();

                for (int i = 0; i < captureResult.size() ; i++) {
                    captureText += captureResult.get(i);
                }
                speechText.setText(originText+" " + captureText);

//                confidenceText.setText(captureText);
                classifyMBTI(captureText);
                speechRecognizer.startListening(intent);
            }

            @Override
            public void onPartialResults(Bundle partialResults) {
//                ArrayList<String> captureResult = partialResults.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
//                String originText = speechText.getText().toString();
//                String resultText="";
//                for (int i = 0; i < captureResult.size() ; i++) {
//                    resultText += captureResult.get(i);
//                }
//
////                confidenceText.setText(captureText);
////                resultText.setText(classifyMBTI(captureText));
//                speechText.setText(originText+" " + resultText);
//                speechRecognizer.startListening(intent);

            }

            @Override
            public void onEvent(int eventType, Bundle params) {

            }
        });
        speechRecognizer.startListening(intent);
    }

    private void endAudioCapture(){
        speechRecognizer.stopListening();
    }

    private List<Integer> wordPieceTokenizer(String questionOrText) {
        List<Integer> tokenIds = new ArrayList<>();
        Pattern p = Pattern.compile("\\w+|\\S");
        Matcher m = p.matcher(questionOrText);
        while (m.find()) {
            String token = m.group().toLowerCase();
            if (mTokenIdMap.containsKey(token)) {
                tokenIds.add(mTokenIdMap.get(token));
            } else {
                for (int i = 0; i < token.length(); i++) {
                    if (mTokenIdMap.containsKey(token.substring(0, token.length() - i - 1))) {
                        tokenIds.add(mTokenIdMap.get(token.substring(0, token.length() - i - 1)));
                        String subToken = token.substring(token.length() - i - 1);
                        int j = 0;
                        while (j < subToken.length()) {
                            if (mTokenIdMap.containsKey("##" + subToken.substring(0, subToken.length() - j))) {
                                tokenIds.add(mTokenIdMap.get("##" + subToken.substring(0, subToken.length() - j)));
                                subToken = subToken.substring(subToken.length() - j);
                                j = subToken.length() - j;
                            }
                            j++;
                        }
                    }
                }
            }
        }
        return tokenIds;
    }

    private List<Integer> tokenizer(String text) {
        List<Integer> rawTokenIds = wordPieceTokenizer(text);
        List<Integer> tokenIds = new ArrayList<>();

        tokenIds.add(mTokenIdMap.get(CLS));
        for (int i = 0; i < Math.min(tokenLimit, rawTokenIds.size()); i++) {
            tokenIds.add(rawTokenIds.get(i));
        }
        tokenIds.add(mTokenIdMap.get(SEP));
//        for (int i = 0; i < MODEL_INPUT_LENGTH - EXTRA_ID_NUM - rawTokenIds.size(); i++) {
//            tokenIds.add(mTokenIdMap.get(PAD));
//        }
        return tokenIds;
    }


    public void classifyMBTI(String text) {
        try {
            List<Integer> tokenIds = tokenizer(text);
            Log.i(TAG, "Tokenizing Successful!");
            IntBuffer inTensorBuffer = Tensor.allocateIntBuffer(tokenIds.size());
            for (Integer n : tokenIds) {
                inTensorBuffer.put(n);
            }
            Log.i(TAG, "Input Buffer Created");

            Tensor inTensor = Tensor.fromBlob(inTensorBuffer, new long[]{1, tokenIds.size()});
            Log.i(TAG, "Input Tensor Created");

            Tensor outTensor = module.forward(IValue.from(inTensor)).toTensor();
            Log.i(TAG, "Module Forward Successful!");

            float[] outputProb = outTensor.getDataAsFloatArray();
            for (int i=0; i<16; i++){
                Log.i(TAG, "Output Prob" + String.valueOf(i) + "th: " + String.valueOf(outputProb[i]));
            }
//            int maxIndex = argmax(outputProb);
            int[] sortedIndex = argsort(outputProb);

//            resultText.setText(MBTI[maxIndex]);
            resultText.setText(MBTI[sortedIndex[0]]);
            confidenceText.setText("Confidence: " + String.valueOf(outputProb[sortedIndex[0]]));

            drawBarChart(barChart, sortedIndex, outputProb);
        } catch (Exception e) {
            Log.e(TAG, "Error during Classification", e);
        }
    }

    private int[] argsort(float[] array) {
        int[] sorted = new int[numLabels];
        for (int i=1; i<numLabels; i++){
            sorted[i] = i;
            for (int j=i-1; j>=0; j--){
                if (array[sorted[j+1]]>array[sorted[j]]){
                    int tmp = sorted[j];
                    sorted[j] = sorted[j+1];
                    sorted[j+1] = tmp;
                }
            }
        }
        return sorted;

//        int maxIdx = 0;
//        double maxVal = -Double.MAX_VALUE;
//        for (int j = 0; j < array.length; j++) {
//            if (array[j] > maxVal) {
//                maxVal = array[j];
//                maxIdx = j;
//            }
//        }
//        return maxIdx;
    }

    private void drawBarChart(BarChart barChart, int[] idx, float[] data) {
        BarDataSet[] barDataList = new BarDataSet[numPlot];
        for (int i = 0; i < numPlot; i++){
            ArrayList<BarEntry> entryList = new ArrayList<>();
            entryList.add(new BarEntry(i, data[idx[i]]));
            barDataList[i] = new BarDataSet(entryList, MBTI[idx[i]]);
            barDataList[i].setColor(colorList[i]);
        }

        BarData barData = new BarData(barDataList[0], barDataList[1], barDataList[2], barDataList[3], barDataList[4]);
        barChart.setData(barData);

        Description description = new Description();
        description.setText("");
        barChart.setDescription(description);

        // Enable chart interaction
        barChart.setTouchEnabled(true);
        barChart.setDragEnabled(true);
        barChart.setScaleEnabled(true);

        // Customize x-axis (hide labels, if needed)
        barChart.getXAxis().setDrawLabels(true); // Set to false to hide x-axis labels

        // Refresh chart
        barChart.invalidate();
    }

}