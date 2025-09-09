class RegisterRequestModel {
  String email;
  String password;
  String confirmPassword;

  RegisterRequestModel({
    required this.email,
    required this.password,
    required this.confirmPassword,
  });

  Map<String, dynamic> toJson() {
    return {
      'email': email,
      'password': password,
      'confirmPassword': confirmPassword,
    };
  }
}
