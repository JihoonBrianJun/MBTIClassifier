package com.mbticlassifierapp;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

public class MainActivity extends AppCompatActivity {

    String TAG = "MainPage";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.i(TAG, "Successfully Created activity_main");
    }

    public void onStartSpeechActivity(View view){
        Intent intent = new Intent(this, SpeechActivity.class);
        startActivity(intent);
        Log.i(TAG, "Entered into InertialActivity");
    }

}