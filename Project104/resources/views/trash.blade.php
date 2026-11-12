@extends('layouts/master')

@section('content')


<main id="main" class="main">

    @csrf
    <div class="pagetitle">
        <h1>Trash</h1>
    </div><!-- End Page Title -->
    <section class="section dashboard">

        <div class="row">
            <!-- Left side columns -->
            <div class="col-lg-12">

                <div class="row">

                    <div class="col-xxl-8 col-md-8">
                        <div class="card-dashboard text-end py-2">

                        </div>
                    </div>

                    <div class="col-xxl-4 col-md-4">
                        <div class="card-dashboard text-end">

                        </div>
                    </div>

                    <!-- Editable table -->
                    @if (count($sales) > 0)
                    
                    <div class="card pt-4 mt-3">
                        <div class="card-body">
                            <table class="table">
                                <thead class="thead-dark" style="background-color: #212529; color:aliceblue;">
                                    <tr>
                                        <th scope="col"></th>
                                        <th scope="col">SO</th>
                                        <th scope="col">Customer</th>
                                        <th scope="col">Owner</th>
                                        <th scope="col">SO Locked Date</th>
                                        <th scope="col">Action</th>

                                    </tr>
                                </thead>
                                <tbody>
                                    @foreach ($sales as $sale)
                                    <tr>
                                        <th scope="row">{{ $sale->id }}</th>
                                        <td>{{ $sale->so }}</td>
                                        <td>{{ $sale->customer_name }}</td>
                                        <td>{{ $sale->owner}}</td>
                                        <td>{{ $sale->so_locked_date}}</td>
                                        <td class="fs-4">
                                            <span class="restorebtn" style="color: #0ea5e9;cursor: pointer;" data-id="{{ $sale->id }}"><i class="bi bi-box-arrow-up"></i></span>
                                        </td>

                                    </tr>

                                    @endforeach
                                </tbody>
                            </table>
                        </div>
                    </div>
                    @else
                    <h1> Tash Empty</h1>
                    @endif
                </div>

            </div><!-- End Left side columns -->
        </div>

    </section>

</main><!-- End #main -->

<script src="https://code.jquery.com/jquery-3.7.0.min.js" integrity="sha256-2Pmvv0kuTBOenSvLm6bvfBSSHrUJ+3A7x6P5Ebd07/g=" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>

<script>   
    $(document).ready(function(){
    $(".restorebtn").click(
        function() {
            Swal.fire({
                title: 'Are you sure?',
                text: "Restoring the Sale",
                icon: 'question',
                showCancelButton: true,
                confirmButtonColor: '#3085d6',
                cancelButtonColor: '',
                confirmButtonText: 'Yes, Restore it!'
            }).then((result) => {
                if (result.isConfirmed) {
                    //data to submit
                    var data = {
                        id:$(this).data("id")
                    }
                    $.ajax({
                        url: "/restore-sale",
                        type: "POST",
                        headers: {
                            "X-CSRF-TOKEN": $('meta[name="csrf-token"]').attr(
                                "content"
                            ),
                            Accept: "application/json",
                        },
                        data:data,
                        success: function(response) {
                            location.reload("/");
                        },
                        error: function() {
                            Swal.fire(
                                'Error!',
                                'Restore Sale Error',
                                'error'
                            )
                        },
                    });
                }
            })
        })
    });
</script>
@endsection('content')