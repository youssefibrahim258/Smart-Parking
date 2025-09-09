import 'package:flutter/material.dart';
import 'package:project_parking_app/model/auth/login_request_model.dart';
import 'package:project_parking_app/model/auth/user_model.dart';
import 'package:project_parking_app/services/auth_services.dart';

class AuthViewModel extends ChangeNotifier {
  final AuthServices _authServices = AuthServices();
  UserModel? user;
  bool isLoading = false;
  String? error;

  Future<bool> login(String email, String password) async {
    isLoading = true;
    error = null;
    notifyListeners();
    final result = await _authServices
        .login(LoginRequestModel(email: email, password: password));
    if (result.startsWith('Error')) {
      error = result;
      isLoading = false;
      notifyListeners();
      return false;
    } else {
      // Parse user data if needed
      user = UserModel(email: email, token: result);
      isLoading = false;
      notifyListeners();
      return true;
    }
  }

  Future<bool> register(String email, String password) async {
    isLoading = true;
    error = null;
    notifyListeners();
    final result = await _authServices.register(email, password);
    if (result.startsWith('Error')) {
      error = result;
      isLoading = false;
      notifyListeners();
      return false;
    } else {
      user = UserModel(email: email, token: result);
      isLoading = false;
      notifyListeners();
      return true;
    }
  }

  Future<void> logout() async {
    await _authServices.logout();
    user = null;
    notifyListeners();
  }
}
