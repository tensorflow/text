<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="utf-8">
  <meta content="width=device-width, initial-scale=1.0" name="viewport">

  <title>Login - Dashbaord</title>
  <meta content="" name="description">
  <meta content="" name="keywords">

  <!-- Favicons -->
  <link href="assets/img/favicon.png" rel="icon">
  <link href="assets/img/apple-touch-icon.png" rel="apple-touch-icon">

  <!-- Google Fonts -->
  <link href="https://fonts.gstatic.com" rel="preconnect">

  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,700;1,400;1,500&display=swap" rel="stylesheet">

  <!-- Vendor CSS Files -->
  <link href="assets/bootstrap/css/bootstrap.min.css" rel="stylesheet">
  <link href="assets/bootstrap-icons/bootstrap-icons.css" rel="stylesheet">

  <link href="assets/css/mdb.min.css" rel="stylesheet">
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" rel="stylesheet">

  <script src="assets/js/jquery.min.js"></script>
  <link href="assets/css/style.css" rel="stylesheet">


</head>

<body class="login-bg">

  <main>
    <div class="container-fluid">

      <section class="section register min-vh-100 d-flex flex-column align-items-center justify-content-center py-4">
        <div class="container-fluid">
          <div class="row justify-content-center">
            <div class="col-lg-10 col-md-10 flex-column align-items-center justify-content-center">



              <div class="mb-3">

                <div class="card-body login-form relative">
              <div class="row">
                <div class="col">
                  <div class="py-4">
                  <a href="{{ route('index') }}" class="logo">
                    <img src="assets/img/logo.png" alt=""> </a> <span class="head">Daily Manager</span></div>

                   <div class="justify-content-center"><img src="assets/img/login-img.png" class="ml-5 mt-5 pl-5 vert-move "></div>
                  </div>
                  <div class="col p-5 pl-8">
                    <div class="pt-8 pb-2">
                      <h2 class="main-title pb-0">Hello,<br>
                        Welcome <span class="text-primary">back</span></h2>

                    </div>
                    <form class="row needs-validation" action="{{ route('login') }}" method="post">
                        @method('post')
                        @csrf
                      <div class="col-md-12 mb-4">
                            <div class="card-form">
                                <div class="md-form">
                                    <input type="text" id="email" name="email" class="form-control">
                                    <label for="email" class="text-primary fw-bold">Email</label>
                                </div>
                                <div class="md-form">
                                    <input type="password" id="password" name="password" class="form-control">
                                    <label for="password" class="text-primary fw-bold">Password</label>
                                </div>
                                <div class="md-form">
                                  <div class="form-group float-start">
                                    <input type="checkbox" id="html">
                                    <label style="top:0px !important; color: #1db3db;" for="html">Remember Me</label>
                                  </div>
                                  <div class="form-group float-end">
                                    <p><a href="#">Forgot password?</a> </p>
                                  </div>
                                </div>
                                <div class="clearfix"></div>
                                <div class="text-center">
                                    <button class="btn btn-default waves-effect waves-light col-lg-12 rounded">Login</button>
                                </div>
                            </div>
                        </div>
                    </form>


                  </div>
                </div>



                <div class="version"> © 2023, Powered by <a href="#">Immensa</a> | V.1.0.0  </div>

                </div>
              </div>



            </div>
          </div>
        </div>

      </section>

    </div>
  </main><!-- End #main -->

  <!-- Vendor JS Files -->
  <script src="assets/bootstrap/js/bootstrap.bundle.min.js"></script>



  <script src="assets/js/mdb.min.js"></script>


  <!-- Template Main JS File -->
  <script src="assets/js/main.js"></script>

</body>

</html>

<script src="https://code.jquery.com/jquery-3.7.0.min.js"
integrity="sha256-2Pmvv0kuTBOenSvLm6bvfBSSHrUJ+3A7x6P5Ebd07/g=" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>

@if ($errors->any())
<script>
    const showSweetAlert = () => {
        Swal.fire({
            toast: true,
            icon: '{{ session('success') ? 'success' : 'error' }}',
            title: '{{ session('success') ? 'Success' : 'Error' }}',
            animation: false,
            position: 'top',
            showConfirmButton: false,
            timer: 3000,
            timerProgressBar: true,
            didOpen: (toast) => {
                toast.addEventListener('mouseenter', Swal.stopTimer)
                toast.addEventListener('mouseleave', Swal.resumeTimer)
            }
        });
    };
    showSweetAlert();
</script>
@endif
