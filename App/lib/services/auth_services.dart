import 'package:project_parking_app/model/auth/login_request_model.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class AuthServices {
  String baseUrl = "https://dummyapi.io/api"; // يمكنك استبدالها لاحقًا
  String loginEndPoint = "/login";
  String logoutEndPoint = "/logout";
  String registerEndPoint = "/register";

  // User login method
  Future<String> login(LoginRequestModel loginRequestModel) async {
    try {
      final Uri uri = Uri.parse(baseUrl + loginEndPoint);
      final response = await http.post(
        uri,
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode(loginRequestModel.toJson()),
      );
      if (response.statusCode == 200) {
        // Assuming the response body contains a token or user data
        final responseBody = response.body;
        return responseBody; // Return the token or any relevant data
      } else {
        // Handle error response
        return "Error: ${response.statusCode} - ${response.body}";
      }
    } catch (e) {
      return "Error: $e";
    }
  }

  // User logout method
  Future<void> logout() async {
    // يمكنك إضافة منطق حذف التوكن أو استدعاء endpoint حقيقي لاحقًا
    // مثال:
    // final Uri uri = Uri.parse(baseUrl + logoutEndPoint);
    // await http.post(uri);
  }

  // User registration method
  Future<String> register(String email, String password) async {
    try {
      final Uri uri = Uri.parse(baseUrl + registerEndPoint);
      final response = await http.post(
        uri,
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'email': email,
          'password': password,
        }),
      );
      if (response.statusCode == 200 || response.statusCode == 201) {
        // Assuming the response body contains a token or user data
        final responseBody = response.body;
        return responseBody;
      } else {
        return "Error: ${response.statusCode} - ${response.body}";
      }
    } catch (e) {
      return "Error: $e";
    }
  }
}
