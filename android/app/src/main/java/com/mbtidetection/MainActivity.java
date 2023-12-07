package com.mbtidetection;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.location.Location;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.mbtidetection.R;

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
