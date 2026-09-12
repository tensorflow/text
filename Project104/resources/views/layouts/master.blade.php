<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">

<head>
    <meta charset="utf-8">
    <meta content="width=device-width, initial-scale=1.0" name="viewport">
    <meta name="csrf-token" content="{{ csrf_token() }}">

    <title>Dashboard</title>
    <meta content="" name="description">
    <meta content="" name="keywords">

    <!-- Favicons -->
    <link href="assets/img/favicon.png" rel="icon">
    <link href="assets/img/apple-touch-icon.png" rel="apple-touch-icon">

    <!-- Google Fonts -->
    <link href="https://fonts.gstatic.com" rel="preconnect">
    <link href="https://fonts.googleapis.com/css?family=Open+Sans:300,300i,400,400i,600,600i,700,700i|Nunito:300,300i,400,400i,600,600i,700,700i|Poppins:300,300i,400,400i,500,500i,600,600i,700,700i" rel="stylesheet">


    <link href="assets/bootstrap/css/bootstrap.min.css" rel="stylesheet">
    <link href="assets/bootstrap-icons/bootstrap-icons.css" rel="stylesheet">

    <link href="assets/css/jquery.mCustomScrollbar.min.css" rel="stylesheet">

    <link href="assets/css/style.css" rel="stylesheet">

    <script src="assets/js/jquery-2.1.4.min.js"></script>

    <!-- <script>
        $(document).ready(function() {
            $(".content").mCustomScrollbar({
                theme: "dark",
                scrollButtons: {
                    scrollType: "stepped"
                },
                live: "on"
            });
        });
    </script> -->

</head>

<body>

    <div class="header">
        <i class="bi bi-list toggle-sidebar-btn"></i>
    </div>
    <!-- ======= Sidebar ======= -->
    <aside id="sidebar" class="sidebar">

        <div class="d-flex align-items-center justify-content-between">
            <a href="{{ route('index') }}" class="logo d-flex align-items-center">
                <img src="assets/img/logo.png" alt="">

            </a>

        </div><!-- End Logo -->
        <?php

        use Illuminate\Support\Facades\Route;
        ?>
        <ul class="sidebar-nav" id="sidebar-nav">

            <li class="nav-item">
                <a class="nav-link {{ (Route::current()->getName() == 'index') ? '' : 'collapsed' }} " href="{{ route('index') }}">
                    <i class="bi bi-grid"></i>
                    <span>Dashboard</span>
                </a>
            </li><!-- End Dashboard Nav -->
            @auth
            @if (auth()->user()->role == 'superAdmin')
            <li class="nav-item">
                <a class="nav-link {{ (Route::current()->getName() == 'userindex') ? '' : 'collapsed' }}" href="{{ route('userindex') }}">
                    <i class="bi bi-person"></i>
                    <span>User</span>
                </a>
            </li>
            @endif
            @endauth
        </ul>
    </aside><!-- End Sidebar-->
    <div class="main-body">


        @yield('content')

        @yield('scripts')



    </div>

    <nav class="sidebar-width d-none d-md-block bg-white sidebar-right">

        <div class="sidebar-sticky">
            <ul class="nav flex-column">

                <li class="nav-item dropdown pe-3">

                    <a class="nav-link nav-profile d-flex align-items-center pe-0" href="#" data-bs-toggle="dropdown">
                        <img src="assets/img/user-img.png" alt="Profile" class="rounded-circle user-img">
                    </a><!-- End Profile Iamge Icon -->

                    <ul class="dropdown-menu dropdown-menu-end dropdown-menu-arrow profile">
                        <li class="dropdown-header">
                            <h6>Kevin Anderson</h6>
                            <span>Project Manager</span>
                        </li>
                        <li>
                            <hr class="dropdown-divider">
                        </li>

                        <li>
                            <a class="dropdown-item d-flex align-items-center" href="#">
                                <i class="bi bi-person"></i>
                                <span>My Profile</span>
                            </a>
                        </li>
                        <li>
                            <hr class="dropdown-divider">
                        </li>

                        <li>
                            <a class="dropdown-item d-flex align-items-center" href="#">
                                <i class="bi bi-gear"></i>
                                <span>Account Settings</span>
                            </a>
                        </li>
                        <li>
                            <hr class="dropdown-divider">
                        </li>


                        <li>
                            <hr class="dropdown-divider">
                        </li>

                        <li>
                            <a action="#" method="post" class="dropdown-item d-flex align-items-center" href="#">
                                <form action="{{ route('logout') }}" method="post">
                                    @method('post')
                                    @csrf
                                    <i class="bi bi-box-arrow-right"></i>

                                    <button type="submit"> <span>Log Out</span></button>
                                </form>
                            </a>
                        </li>

                    </ul><!-- End Profile Dropdown Items -->
                    @auth
                    @if (auth()->user()->role == 'superAdmin')
                    <li class="nav-item pe-3"> 
                    <a class="nav-link nav-profile d-flex align-items-center pe-0" href="{{ route('trash') }}">
                        <div class="rounded-circle" style="width: 42px;height: 42px;line-height: 1.3;text-align: center;color: #373737;box-shadow: 0px 2px 10px 0px #797676bf;"><i class="bi fs-2 bi-recycle"></i></div>
                    </a></li>
                    @endif
                    @endauth
                </li><!-- End Profile Nav -->


            </ul>
        </div>
    </nav>



    <!-- End Footer -->

    <a href="#" class="back-to-top d-flex align-items-center justify-content-center"><i class="bi bi-arrow-up-short"></i></a>

    <!-- Vendor JS Files -->

    <script src="assets/bootstrap/js/bootstrap.bundle.min.js"></script>



    <!-- Template Main JS File -->
    <script src="assets/js/main.js"></script>


    <script>
        const mouseWheel = document.querySelector('.inner-wrapper');

        mouseWheel.addEventListener('wheel', function(e) {
            const race = 15; // How many pixels to scroll

            if (e.deltaY > 0) // Scroll right
                mouseWheel.scrollLeft += race;
            else // Scroll left
                mouseWheel.scrollLeft -= race;
            e.preventDefault();
        });
    </script>


</body>

</html>