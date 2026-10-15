@extends('layouts/master')

@section('content')


<main id="main" class="main">

    @csrf
    <div class="pagetitle">
        <h1>Users</h1>
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

                            <button type="button" class="modal_btn btn btn-info me-3 cursor-pointer"><i class="bi bi-plus-lg"></i>Add
                                New</button>
                            <a class="btn btn-info" href="/users"><i>Cancel</i></a>
                            {{-- <button type="button" class="btn btn-info" href="#" role="button">Cancel</button> --}}

                        </div>
                    </div>

                    <!-- Editable table -->
                    @if (count($users) > 0)
                    
                    <div class="card pt-4 mt-3">
                        <div class="card-body">
                            <table class="table">
                                <thead class="thead-dark" style="background-color: #212529; color:aliceblue;">
                                    <tr>
                                        <th scope="col"></th>
                                        <th scope="col">Name</th>
                                        <th scope="col">Email</th>
                                        <th scope="col">Role</th>
                                        <th scope="col">Action</th>

                                    </tr>
                                </thead>
                                <tbody>
                                    @foreach ($users as $user)
                                    <tr>
                                        <th scope="row">{{$user->id}}</th>
                                        <td>{{ $user->name }}</td>
                                        <td>{{ $user->email }}</td>
                                        <td>{{ $user->role}}</td>
                                        <td class="fs-4">
                                            <span class="updatebtn" style="color: #0ea5e9;cursor: pointer;" data-name="{{ $user->name }}" data-email="{{ $user->email }}" data-role="{{ $user->role }}" data-id="{{ $user->id }}"><i class="bi bi-vector-pen"></i></span>
                                            <span class="user_delete" style="color: #dc2626;cursor: pointer;" data-user="{{ $user->id }}"><i class="bi bi-trash"></i></span>
                                        </td>

                                    </tr>

                                    @endforeach
                                </tbody>
                            </table>
                        </div>
                    </div>
                    @else
                    <h1> No User</h1>
                    @endif
                </div>

            </div><!-- End Left side columns -->
        </div>

    </section>

</main><!-- End #main -->
<style>

.modal_box,.modal_box_update{
	position: relative;
	width: 100%;
	height: 100%;
	z-index: 999;
	display: none;
}

.modal_bg_shadow{
	position: fixed;
	top: 0;
	left: 0;
	background: #000;
	opacity: 0.5;
	width: 100%;
	height: 100%;
	z-index: -1;
}

.modal_box_wrap{
	width: 550px;
	height: 420px;
	background: #fff;
	position: absolute;
	top: 50%;
	left: 50%;
	transform: translate(-50%,-50%);
	border-radius: 5px;
}
 
.modal_box_wrap .modal_close{
	position: absolute;
	top: -25px;
	right: -25px;
	width: 50px;
	height: 50px;
	background: #363d4e;
	border-radius: 50%;
	cursor: pointer;
}

.modal_box_wrap .modal_close:before,
.modal_box_wrap .modal_close:after{
	content: "";
	position: absolute;
	top: 25px;
	left: 13px;
	width: 25px;
	height: 2px;
	background: #fff;
}

.modal_box_wrap .modal_close:before{
	transform: rotate(45deg);
}

.modal_box_wrap .modal_close:after{
	transform: rotate(130deg);
}

.modal_box_wrap .modal_header{
	padding: 20px;
	border-bottom: 1px solid #e0e0e0;
	height: 60px;
	font-size: 22px;
}	


.modal_box_wrap .modal_body{
	padding: 20px;
	border-bottom: 1px solid #e0e0e0;
	font-size: 14px;
	line-height: 21px;
}

.modal_box_wrap .modal_footer{
	padding: 20px;
	height: 60px;
}

.modal_footer .modal_btn_grp{
	display: flex;
	justify-content: flex-end;
	align-items: center;
	height: 100%;
}

.modal_footer .modal_btn_grp .btn{
	width: 100px;
	padding: 10px;
	border-radius: 5px;
	text-align: center;
	cursor: pointer;
}

.modal_footer .modal_btn_grp .btn.btn_confirm{
	margin-left: 10px;
	background: #363d4e;
	color: #fff;
}
.modal_footer .modal_btn_grp .btn.btn_update_confirm{
	margin-left: 10px;
	background: #363d4e;
	color: #fff;
}

.modal_footer .modal_btn_grp .btn.btn_cancel{
	border: 1px solid #363d4e;
	color: #363d4e;
}

.modal_footer .modal_btn_grp .btn.btn_cancel:hover{
	background: #363d4e;
	color: #fff;
}

.modal_footer .modal_btn_grp .btn:hover{
	background: #7b8d8f;
}

.modal_box.active,.modal_box_update.active{
	display: block;
}</style>
{{-- Add New Modal --}}
    <div class="modal_box">
		<div class="modal_bg_shadow"></div>
		<div class="modal_box_wrap">
			<div class="modal_close"></div>
			<div class="modal_header">
				Add New User
			</div>
			<div class="modal_body">
				<form>
                        <div>
                        <label for="inputName">Name</label>
                        <input type="text" class="form-control" id="inputName" placeholder="Enter name">
                        <small class="text-danger addemail"></small>
                        </div>
                        <div>
                        <label for="inputEmail" class="mt-3">Email address</label>
                        <input type="email" class="form-control" id="inputEmail" aria-describedby="emailHelp" placeholder="Enter email">
                        <small class="text-danger addname"></small>                    
                        </div>
                        <div>
                        <label for="inputRole" class="mt-3">Role</label>
                        <select class="form-control" id="inputRole">
                            <option value="superAdmin">Super Admin</option>
                            <option value="admin">Admin</option>
                            <option value="customer" selected>Customer</option>
                        </select>
                        </div>
                </form>
			</div>
			<div class="modal_footer">
				<div class="modal_btn_grp">
					<div class="btn btn_cancel">Cancel</div>
					<div class="btn btn_confirm">Submit</div>
				</div>
			</div>
		</div>
	</div>
    {{-- Update Modal --}}
    <div class="modal_box_update">
		<div class="modal_bg_shadow"></div>
		<div class="modal_box_wrap">
			<div class="modal_close"></div>
			<div class="modal_header">
				Update User
			</div>
			<div class="modal_body">
				<form>
                        <div>
                        <input type="hidden"  id="updateId">
                        <label for="updateName">Name</label>
                        <input type="text" class="form-control" id="updateName" placeholder="Enter name">
                        <small class="text-danger updateemail"></small>
                        </div>
                        <div>
                        <label for="updateEmail" class="mt-3">Email address</label>
                        <input type="email" class="form-control" id="updateEmail" aria-describedby="emailHelp" placeholder="Enter email">
                        <small class="text-danger updatename"></small>                    
                        </div>
                        <div>
                        <label for="updateRole" class="mt-3">Role</label>
                        <select class="form-control" id="updateRole">
                            <option value="superAdmin">Super Admin</option>
                            <option value="admin">Admin</option>
                            <option value="customer" selected>Customer</option>
                        </select>
                        </div>
                </form>
			</div>
			<div class="modal_footer">
				<div class="modal_btn_grp">
					<div class="btn btn_cancel">Cancel</div>
					<div class="btn btn_update_confirm">Submit</div>
				</div>
			</div>
		</div>
	</div>

<script src="https://code.jquery.com/jquery-3.7.0.min.js" integrity="sha256-2Pmvv0kuTBOenSvLm6bvfBSSHrUJ+3A7x6P5Ebd07/g=" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>

<script>   
function ClearData(){
    $("#inputName").val('');
    $("#inputEmail").val('');
    $("#updateName").val('');
    $("#updateEmail").val('');
    $('.addemail').text('');
    $('.addname').text('');
    $('.updateemail').text('');
    $('.updatename').text('');
}
    $(document).ready(function(){

			$(".modal_btn").click(function(){
				$(".modal_box").addClass("active");
			});

			$(".modal_close").click(function(){
				$(".modal_box").removeClass("active");
                $(".modal_box_update").removeClass("active");
                ClearData();
                
			});
			$(".btn_cancel").click(function(){
				$(".modal_box").removeClass("active");
                $(".modal_box_update").removeClass("active");
                ClearData();
			});

            $(".updatebtn").click(function(){
                $("#updateId").val($(this).data('id')),
                $("#updateName").val($(this).data('name')),
                $("#updateEmail").val($(this).data('email')),
                $('#updateRole option[value="'+$(this).data('role')+'"]').prop('selected',true)
                $(".modal_box_update").addClass("active");
            })
            {{-- Add User --}}
            $(".btn_confirm").click(function(){
                var inputData={
                    name:$("#inputName").val(),
                    email:$("#inputEmail").val(),
                    role:$("#inputRole option:selected").val()
                }
                $.ajax({
                    url:"/add-new-user",
                    type:"POST",
                    headers: {
                            "X-CSRF-TOKEN": $('meta[name="csrf-token"]').attr(
                                "content"
                            ),
                            Accept: "application/json",
                    },
                    data:inputData,
                    success: function(response) {
                        location.reload();
                    },
                    error: function(xhr, ajaxOptions, thrownError) {
                        if(xhr.status==422){
                            $('.addemail').text(xhr.responseJSON.errors.name?xhr.responseJSON.errors.name[0]:'')
                            $('.addname').text(xhr.responseJSON.errors.email?xhr.responseJSON.errors.email[0]:'')
                        }else{
                            $(".modal_close").trigger('click');
                            Swal.fire(
                                'Error!',
                                'Add User Error',
                                'error'
                            )
                        }
                        },
                })

            });

        {{-- Update User --}}
        $(".btn_update_confirm").click(function(){
                var inputData={
                    name:$("#updateName").val(),
                    email:$("#updateEmail").val(),
                    role:$("#updateRole option:selected").val()
                }
                $.ajax({
                    url:"/update-user/"+$("#updateId").val(),
                    type:"POST",
                    headers: {
                            "X-CSRF-TOKEN": $('meta[name="csrf-token"]').attr(
                                "content"
                            ),
                            Accept: "application/json",
                    },
                    data:inputData,
                    success: function(response) {
                        location.reload();
                    },
                    error: function(xhr, ajaxOptions, thrownError) {
                        if(xhr.status==422){
                            $('.updateemail').text(xhr.responseJSON.errors.name?xhr.responseJSON.errors.name[0]:'')
                            $('.updatename').text(xhr.responseJSON.errors.email?xhr.responseJSON.errors.email[0]:'')
                        }else{
                            $(".modal_close").trigger('click');
                            Swal.fire(
                                'Error!',
                                'Add User Error',
                                'error'
                            )
                        }
                        },
                })

            });


	});
    $(".user_delete").click(
        function() {
            Swal.fire({
                title: 'Are you sure?',
                text: "You won't be able to revert this!",
                icon: 'question',
                showCancelButton: true,
                confirmButtonColor: '#d33',
                cancelButtonColor: '#3085d6',
                confirmButtonText: 'Yes, delete it!'
            }).then((result) => {
                if (result.isConfirmed) {
                    //data to submit
                    var data = $(this).data("user");
                    console.log(data);
                    $.ajax({
                        url: "/delete-user/" + data,
                        type: "DELETE",
                        headers: {
                            "X-CSRF-TOKEN": $('meta[name="csrf-token"]').attr(
                                "content"
                            ),
                            Accept: "application/json",
                        },
                        success: function(response) {
                            location.reload();
                        },
                        error: function() {
                            Swal.fire(
                                'Error!',
                                'Delete User Error',
                                'error'
                            )
                        },
                    });
                }
            })
        }
    );
</script>
@endsection('content')