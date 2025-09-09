import 'package:flutter/material.dart';
import 'dart:async';

class GlobalTimerProvider extends ChangeNotifier {
  int seconds = 0;
  Timer? _timer;
  bool _isRunning = false;

  void start() {
    if (_isRunning) return;
    _isRunning = true;
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      seconds++;
      notifyListeners();
    });
  }

  void stop() {
    _timer?.cancel();
    _isRunning = false;
  }

  void reset() {
    stop();
    seconds = 0;
    notifyListeners();
  }
}
