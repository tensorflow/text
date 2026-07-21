@extends('layouts/master')

@section('content')

<?php $cantarget = 'false'; ?>
<?php $canall = 'true'; ?>
<?php $superAdmin = auth()->user()->role == 'superAdmin' ? 'true' : 'false'; ?>
@auth

@if (auth()->user()->role == 'admin' || auth()->user()->role == 'superAdmin')
<?php $cantarget = 'true'; ?>
@endif
@endauth
<?php
function Colorizer(?string $actualDate, ?string $actualQT, ?string $targetDate, ?string $targetQT): string
{
    $actual_date = new DateTime($actualDate);
    $target_date = new DateTime($targetDate);
    if ($actualDate == null or $actualQT == null or $targetDate == null or $targetQT == null) {
        return 'defaultblack';
    }
    if ($actual_date < $target_date) {
        if ($actualQT == $targetQT) {
            return 'equalgreen';
        } else {
            return 'beforeblue';
        }
    } elseif ($actual_date == $target_date) {
        return 'equalgreen';
    } elseif ($actual_date > $target_date) {
        return 'latered';
    }
}
function NullCheck(?string $value): string
{
    if (is_null($value)) {
        return 'N/A';
    } else {
        return $value;
    }
}
?>

<style>
    .tbl-cell {
        height: 111px;
    }

    .date-col {
        width: 120px;
    }

    .beforeblue {
        color: #fff !important;
        background: #0284c7 !important;
        padding: 10px 5px;
        display: block;
    }

    .equalgreen {
        color: #fff !important;
        background: #38be69 !important;
        padding: 10px 5px;
        display: block;
    }

    .latered {
        color: #fff !important;
        background: #e84d4d !important;
        padding: 10px 5px;
        display: block;
    }

    .defaultblack {
        color: #000000 !important;
        padding: 10px 5px;
        display: block;

    }
</style>
<main id="main" class="main">

    @csrf
    <div class="pagetitle">
        <h1>Dashboard</h1>
    </div><!-- End Page Title -->
    <section class="section dashboard">

        <div class="row">
            <!-- Left side columns -->
            <div class="col-lg-12">

                <div class="row">

                    <div class="col-xxl-8 col-md-8">
                        <div class="card-dashboard text-end py-2">

                            <span class="badge bg-success last_update_date">
                                <?php echo date('d/m/Y', strtotime($last_update_date)); ?>
                            </span>
                            <span class="last_update_duration">
                                Last update:
                                {{ $last_update_duration }}
                            </span>
                        </div>
                    </div>

                    <div class="col-xxl-4 col-md-4">
                        <div class="card-dashboard text-end">

                            <a class="btn btn-info me-3" href="/add-new-sale-ui"><i class="bi bi-plus-lg"></i>Add
                                New</a>
                            <a class="btn btn-info" href="/"><i>Cancel</i></a>
                            {{-- <button type="button" class="btn btn-info" href="#" role="button">Cancel</button> --}}

                        </div>
                    </div>

                    <!-- Editable table -->
                    @if (count($sales) > 0)
                    <div class="card pt-4 mt-3">

                        <div class="card-body">
                            <div class="element" id="element">

                                <div id="table" class="table-editable first-view outer-wrapper float-start ">

                                    <table class="table pb-0 mb-0">
                                        <thead class="black white-text">
                                            <tr>
                                                <th scope="col" class="sales-bg text-center text-white h5">Sales
                                                </th>
                                            </tr>
                                        </thead>
                                    </table>

                                    <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                        <thead>
                                            <tr class="tbl-cell">
                                                <th class="text-center p-2">Delete</th>
                                                @auth
                                                @if (auth()->user()->role == 'superAdmin')
                                                <th class="text-center">Undo</th>
                                                @endif
                                                @endauth
                                                <th class="text-center">SO</th>
                                                <th class="text-center">Sales</th>
                                                <th class="text-center date-col">PO date</th>
                                                <th class="text-center">Customer Name</th>
                                            </tr>
                                        </thead>

                                        <tbody>
                                            @foreach ($sales as $sa)
                                            <tr class="tbl-cell">
                                                <td class="p-0">
                                                    <button type="button" class="btn btn_delete_sale" data-id="{{ $sa->id }}"><i class="bi bi-trash3 "></i></button>
                                                </td>
                                                @auth
                                                @if (auth()->user()->role == 'superAdmin')
                                                <td class="p-0">
                                                    <button type="button" class="btn btn_undo_sale" data-id="{{ $sa->id }}"><i class="bi bi-arrow-counterclockwise"></i></button>
                                                </td>
                                                @endif
                                                @endauth
                                                <td class="pt-3-half tbl_cell " contenteditable="true" data-id="{{ $sa->id }}" data-col_name="so">
                                                    {{ $sa->so }}
                                                </td>
                                                <td class="pt-3-half tbl_cell " contenteditable="true" data-id="{{ $sa->id }}" data-col_name="salesman">
                                                    {{ $sa->salesman }}
                                                </td>
                                                <td class="pt-3-half tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="po_date">
                                                    {{ $sa->po_date }}
                                                </td>

                                                <td class="pt-3-half tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="customer_name">
                                                    {{ $sa->customer_name }}
                                                </td>
                                            </tr>
                                            @endforeach
                                        </tbody>
                                    </table>
                                </div>

                                <div class="outer-wrapper">
                                    <div class="inner-wrapper">

                                        <div id="table" class="table-editable ">

                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="sales-bg text-center text-white h5 first-col-bg">
                                                        </th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">Country</th>
                                                        <th class="text-center">Type</th>
                                                        <th class="text-center">Strategic Contract</th>
                                                        <th class="text-center">Advance payment</th>
                                                        <th class="text-center">Owner</th>
                                                        <th class="text-center">Client Update
                                                            Date</th>
                                                        <th class="text-center date-col">SO Locked Date</th>
                                                    </tr>
                                                </thead>

                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="pt-3-half tbl_cell" contenteditable="true">

                                                            <select class="form-select no-border border-bottom space_t country_list " aria-label="Default select example" data-id="{{ $sa->id }}">
                                                                @foreach ($countries as $country)
                                                                @if ($sa->country_id == $country->id)
                                                                <option value="{{ $country->id }}" selected>{{ $country->code }}
                                                                </option>
                                                                @else
                                                                <option value="{{ $country->id }}">
                                                                    {{ $country->code }}
                                                                </option>
                                                                @endif
                                                                @endforeach
                                                            </select>
                                                        </td>
                                                        <td class="pt-3-half tbl_cell" contenteditable="{{ $superAdmin }}" data-id="{{ $sa->id }}" data-col_name="type">
                                                            {{ $sa->type }}
                                                        </td>
                                                        <td class="pt-3-half tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="strategic_contract">
                                                            {{ $sa->strategic_contract }}
                                                        </td>
                                                        <td class="pt-3-half tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="advanced_payment">
                                                            {{ $sa->advanced_payment }}
                                                        </td>
                                                        <td class="pt-3-half tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="owner">
                                                            {{ $sa->owner }}
                                                        </td>
                                                        <td class="pt-3-half tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="client_update_date_01">
                                                            {{ $sa->client_update_date_01 }}
                                                        </td>
                                                        <td class="pt-3-half tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="so_locked_date">
                                                            {{ $sa->so_locked_date }}
                                                        </td>
                                                    </tr>
                                                    @endforeach
                                                </tbody>
                                            </table>
                                        </div>
                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="owner-bg text-center text-white h5">
                                                            Owner</th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center date-col">Execution Plan Discharge
                                                            date
                                                        </th>
                                                        <th class="text-center date-col">Client Update</th>
                                                        <th class="text-center date-col">Client KOM date</th>
                                                        <th class="text-center">Part Info From Client</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="pt-3-half tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="execution_plan_discharge_date">
                                                            {{ NullCheck($sa->execution_plan_discharge_date) }}
                                                        </td>
                                                        <td class="pt-3-half tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="client_update_date_02">
                                                            {{ NullCheck($sa->client_update_date_02) }}

                                                        </td>
                                                        <td class="pt-3-half tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="client_kom_date">
                                                            {{ NullCheck($sa->client_kom_date) }}

                                                        </td>
                                                        <td class="p-0">
                                                            <table class="colortableparent" data-tag="1{{ $sa->id }}" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="part_info_from_client_actual_date">
                                                                        <span class="<?php echo Colorizer($sa->part_info_from_client_actual_date, $sa->part_info_from_client_actual_qty, $sa->part_info_from_client_target_date, $sa->part_info_from_client_target_qty); ?>">{{ NullCheck($sa->part_info_from_client_actual_date) }}</span>

                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="part_info_from_client_target_date">
                                                                        <span class="px-2">{{ NullCheck($sa->part_info_from_client_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="part_info_from_client_actual_qty">

                                                                        <span class="<?php echo Colorizer($sa->part_info_from_client_actual_date, $sa->part_info_from_client_actual_qty, $sa->part_info_from_client_target_date, $sa->part_info_from_client_target_qty); ?>">{{ NullCheck($sa->part_info_from_client_actual_qty) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="part_info_from_client_target_qty">
                                                                        <span>{{ NullCheck($sa->part_info_from_client_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach
                                                </tbody>
                                            </table>
                                        </div>

                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="production-bg text-center text-white h5">
                                                            Production
                                                        </th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">Raw Material Outsource</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table class="colortableparent" data-tag="2{{ $sa->id }}" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="raw_matel_outsource_actual_date">

                                                                        <span class="<?php echo Colorizer($sa->raw_matel_outsource_actual_date, $sa->raw_matel_outsource_actual_qty, $sa->raw_matel_outsource_target_date, $sa->raw_matel_outsource_target_qty); ?>">{{ NullCheck($sa->raw_matel_outsource_actual_date) }}</span>

                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="raw_matel_outsource_target_date">
                                                                        <span>{{ NullCheck($sa->raw_matel_outsource_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="raw_matel_outsource_actual_qty">

                                                                        <span class="<?php echo Colorizer($sa->raw_matel_outsource_actual_date, $sa->raw_matel_outsource_actual_qty, $sa->raw_matel_outsource_target_date, $sa->raw_matel_outsource_target_qty); ?>">{{ NullCheck($sa->raw_matel_outsource_actual_qty) }}</span>

                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="raw_matel_outsource_target_qty">
                                                                        <span>{{ NullCheck($sa->raw_matel_outsource_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach
                                                </tbody>
                                            </table>
                                        </div>

                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="engineer-bg text-center text-white h5">
                                                            Engineering
                                                        </th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">Initial Assesmente</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">

                                                            <table class="colortableparent" data-tag="3{{ $sa->id }}" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="initial_assesment_actual_date">

                                                                        <span class="<?php echo Colorizer($sa->initial_assesment_actual_date, $sa->initial_assesment_actual_qty, $sa->initial_assesment_target_date, $sa->initial_assesment_target_qty); ?>">{{ NullCheck($sa->initial_assesment_actual_date) }}</span>

                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="initial_assesment_target_date">
                                                                        <span>{{ NullCheck($sa->initial_assesment_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="initial_assesment_actual_qty">


                                                                        <span class="<?php echo Colorizer($sa->initial_assesment_actual_date, $sa->initial_assesment_actual_qty, $sa->initial_assesment_target_date, $sa->initial_assesment_target_qty); ?>">{{ NullCheck($sa->initial_assesment_actual_qty) }}</span>



                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="initial_assesment_target_qty">
                                                                        <span>{{ NullCheck($sa->initial_assesment_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach
                                                </tbody>
                                            </table>
                                        </div>

                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="owner-bg text-center text-white h5">
                                                            Owner</th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">Pre-mob/mob</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table class="colortableparent" data-tag="4{{ $sa->id }}" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="pre_mob_actual_date">

                                                                        <span class="<?php echo Colorizer($sa->pre_mob_actual_date, $sa->pre_mob_actual_qty, $sa->pre_mob_target_date, $sa->pre_mob_target_qty); ?>">{{ NullCheck($sa->pre_mob_actual_date) }}</span>

                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="pre_mob_target_date">
                                                                        <span>{{ NullCheck($sa->pre_mob_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="pre_mob_actual_qty">

                                                                        <span class="<?php echo Colorizer($sa->pre_mob_actual_date, $sa->pre_mob_actual_qty, $sa->pre_mob_target_date, $sa->pre_mob_target_qty); ?>">{{ NullCheck($sa->pre_mob_actual_qty) }}</span>

                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="pre_mob_target_qty">
                                                                        <span>{{ NullCheck($sa->pre_mob_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach
                                                </tbody>
                                            </table>
                                        </div>

                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="engineer-bg text-center text-white h5">
                                                            Engineering
                                                        </th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center"> Parts Capturing Date</th>
                                                        <th class="text-center"> Design & Engineering date</th>
                                                        <th class="text-center"> DFAM+Build Prep</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    <!-- This is our clonable table line -->

                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table class="colortableparent" data-tag="5{{ $sa->id }}" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="parts_capturing_actual_date">

                                                                        <span class="<?php echo Colorizer($sa->parts_capturing_actual_date, $sa->parts_capturing_actual_qty, $sa->parts_capturing_target_date, $sa->parts_capturing_target_qty); ?>">{{ NullCheck($sa->parts_capturing_actual_date) }}</span>

                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="parts_capturing_target_date">
                                                                        <span>{{ NullCheck($sa->parts_capturing_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="parts_capturing_actual_qty">

                                                                        <span class="<?php echo Colorizer($sa->parts_capturing_actual_date, $sa->parts_capturing_actual_qty, $sa->parts_capturing_target_date, $sa->parts_capturing_target_qty); ?>">{{ NullCheck($sa->parts_capturing_actual_qty) }}</span>

                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="parts_capturing_target_qty">
                                                                        <span>{{ NullCheck($sa->parts_capturing_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>

                                                        <td class="p-0">
                                                            <table class="colortableparent" data-tag="6{{ $sa->id }}" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="design_engineering_actual_date">

                                                                        <span class="<?php echo Colorizer($sa->design_engineering_actual_date, $sa->design_engineering_actual_qty, $sa->design_engineering_target_date, $sa->design_engineering_target_qty); ?>">{{ NullCheck($sa->design_engineering_actual_date) }}</span>

                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="design_engineering_target_date">
                                                                        <span>{{ NullCheck($sa->design_engineering_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="design_engineering_actual_qty">

                                                                        <span class="<?php echo Colorizer($sa->design_engineering_actual_date, $sa->design_engineering_actual_qty, $sa->design_engineering_target_date, $sa->design_engineering_target_qty); ?>">{{ NullCheck($sa->design_engineering_actual_qty) }}</span>

                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="design_engineering_target_qty">
                                                                        <span>{{ NullCheck($sa->design_engineering_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>

                                                        <td class="p-0">
                                                            <table class="colortableparent" data-tag="7{{ $sa->id }}" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="dfam_build_prep_actual_date">

                                                                        <span class="<?php echo Colorizer($sa->dfam_build_prep_actual_date, $sa->dfam_build_prep_actual_qty, $sa->dfam_build_prep_target_date, $sa->dfam_build_prep_target_qty); ?>">{{ NullCheck($sa->dfam_build_prep_actual_date) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="dfam_build_prep_target_date">
                                                                        <span>{{ NullCheck($sa->dfam_build_prep_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="dfam_build_prep_actual_qty">

                                                                        <span class="<?php echo Colorizer($sa->dfam_build_prep_actual_date, $sa->dfam_build_prep_actual_qty, $sa->dfam_build_prep_target_date, $sa->dfam_build_prep_target_qty); ?>">{{ NullCheck($sa->dfam_build_prep_actual_qty) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="dfam_build_prep_target_qty">
                                                                        <span>{{ NullCheck($sa->dfam_build_prep_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach
                                                </tbody>
                                            </table>
                                        </div>

                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="quality-bg text-center text-white h5">
                                                            Quality</th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">ITP + Manufacturing Document</th>
                                                    </tr>
                                                </thead>
                                                <tbody>

                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table class="colortableparent" data-tag="8{{ $sa->id }}" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="itp_manufacturing_document_actual_date">

                                                                        <span class="<?php echo Colorizer($sa->itp_manufacturing_document_actual_date, $sa->itp_manufacturing_document_actual_qty, $sa->itp_manufacturing_document_target_date, $sa->itp_manufacturing_document_target_qty); ?>">{{ NullCheck($sa->itp_manufacturing_document_actual_date) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="itp_manufacturing_document_target_date">
                                                                        <span>{{ NullCheck($sa->itp_manufacturing_document_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="itp_manufacturing_document_actual_qty">

                                                                        <span class="<?php echo Colorizer($sa->itp_manufacturing_document_actual_date, $sa->itp_manufacturing_document_actual_qty, $sa->itp_manufacturing_document_target_date, $sa->itp_manufacturing_document_target_qty); ?>">{{ NullCheck($sa->itp_manufacturing_document_actual_qty) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="itp_manufacturing_document_target_qty">
                                                                        <span>{{ NullCheck($sa->itp_manufacturing_document_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach
                                                </tbody>
                                            </table>
                                        </div>

                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="owner-bg text-center text-white h5">
                                                            Owner</th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">Client Update</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table style="width:100%">

                                                                <tr>
                                                                    <td class="ps-4 tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="client_update_date_03">
                                                                        {{ NullCheck($sa->client_update_date_03) }}
                                                                    </td>
                                                                    {{-- <td class="pt-3-half tbl_cell"
                                                                        contenteditable="true" data-id="{{ $sa->id }}"
                                                                    data-col_name="client_update_date_02">
                                                                    {{ NullCheck($sa->client_update_date_02) }}
                                                        </td> --}}
                                                    </tr>

                                            </table>
                                            </td>





                                            </tr>
                                            @endforeach
                                            </tbody>
                                            </table>
                                        </div>


                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="production-bg text-center text-white h5">
                                                            Production
                                                        </th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">3D Printing</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table class="colortableparent" data-tag="9{{ $sa->id }}" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="_3d_printing_actual_date">

                                                                        <span class="<?php echo Colorizer($sa->_3d_printing_actual_date, $sa->_3d_printing_actual_qty, $sa->_3d_printing_target_date, $sa->_3d_printing_target_qty); ?>">{{ NullCheck($sa->_3d_printing_actual_date) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="_3d_printing_target_date">
                                                                        <span>{{ NullCheck($sa->_3d_printing_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="_3d_printing_actual_qty">

                                                                        <span class="<?php echo Colorizer($sa->_3d_printing_actual_date, $sa->_3d_printing_actual_qty, $sa->_3d_printing_target_date, $sa->_3d_printing_target_qty); ?>">{{ NullCheck($sa->_3d_printing_actual_qty) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="_3d_printing_target_qty">
                                                                        <span>{{ NullCheck($sa->_3d_printing_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach
                                                </tbody>
                                            </table>
                                        </div>


                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="owner-bg text-center text-white h5">
                                                            Owner</th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">Client Update <br>date</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table style="width:100%">

                                                                <tr>
                                                                    <td class="ps-4 tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="client_update_date_04">
                                                                        {{ NullCheck($sa->client_update_date_04) }}
                                                                    </td>
                                                                </tr>

                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach
                                                </tbody>
                                            </table>
                                        </div>

                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="engineer-bg text-center text-white h5">
                                                            Engineering
                                                        </th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">Outsource Production</th>
                                                        <th class="text-center">Post Processing</th>
                                                        <th class="text-center">Assembly</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    <!-- This is our clonable table line -->
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table class="colortableparent" data-tag="1{{ $sa->id }}0" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="outsource_production_actual_date">

                                                                        <span class="<?php echo Colorizer($sa->outsource_production_actual_date, $sa->outsource_production_actual_qty, $sa->outsource_production_target_date, $sa->outsource_production_target_qty); ?>">{{ NullCheck($sa->outsource_production_actual_date) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="outsource_production_target_date">
                                                                        <span>{{ NullCheck($sa->outsource_production_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="outsource_production_actual_qty">

                                                                        <span class="<?php echo Colorizer($sa->outsource_production_actual_date, $sa->outsource_production_actual_qty, $sa->outsource_production_target_date, $sa->outsource_production_target_qty); ?>">{{ NullCheck($sa->outsource_production_actual_qty) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="outsource_production_target_qty">
                                                                        <span>{{ NullCheck($sa->outsource_production_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>
                                                        <td class="p-0">
                                                            <table class="colortableparent" data-tag="1{{ $sa->id }}1" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="post_processing_actual_date">

                                                                        <span class="<?php echo Colorizer($sa->post_processing_actual_date, $sa->post_processing_actual_qty, $sa->post_processing_target_date, $sa->post_processing_target_qty); ?>">{{ NullCheck($sa->post_processing_actual_date) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="post_processing_target_date">
                                                                        <span>{{ NullCheck($sa->post_processing_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="post_processing_actual_qty">

                                                                        <span class="<?php echo Colorizer($sa->post_processing_actual_date, $sa->post_processing_actual_qty, $sa->post_processing_target_date, $sa->post_processing_target_qty); ?>">{{ NullCheck($sa->post_processing_actual_qty) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="post_processing_target_qty">
                                                                        <span>{{ NullCheck($sa->post_processing_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>
                                                        <td class="p-0">
                                                            <table class="colortableparent" data-tag="1{{ $sa->id }}2" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="assembly_actual_date">

                                                                        <span class="<?php echo Colorizer($sa->assembly_actual_date, $sa->assembly_actual_qty, $sa->assembly_target_date, $sa->assembly_target_qty); ?>">{{ NullCheck($sa->assembly_actual_date) }}</span>



                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="assembly_target_date">
                                                                        <span>{{ NullCheck($sa->assembly_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="assembly_actual_qty">

                                                                        <span class="<?php echo Colorizer($sa->assembly_actual_date, $sa->assembly_actual_qty, $sa->assembly_target_date, $sa->assembly_target_qty); ?>">{{ NullCheck($sa->assembly_actual_qty) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="assembly_target_qty">
                                                                        <span>{{ NullCheck($sa->assembly_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach
                                                </tbody>
                                            </table>
                                        </div>

                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="quality-bg text-center text-white h5">
                                                            Quality</th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">QC Testing</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table class="colortableparent" data-tag="1{{ $sa->id }}3" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="qc_testing_actual_date">

                                                                        <span class="<?php echo Colorizer($sa->qc_testing_actual_date, $sa->qc_testing_actual_qty, $sa->qc_testing_target_date, $sa->qc_testing_target_qty); ?>">{{ NullCheck($sa->qc_testing_actual_date) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="qc_testing_target_date">
                                                                        <span>{{ NullCheck($sa->qc_testing_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="qc_testing_actual_qty">

                                                                        <span class="<?php echo Colorizer($sa->qc_testing_actual_date, $sa->qc_testing_actual_qty, $sa->qc_testing_target_date, $sa->qc_testing_target_qty); ?>">{{ NullCheck($sa->qc_testing_actual_qty) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="qc_testing_target_qty">
                                                                        <span>{{ NullCheck($sa->qc_testing_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach
                                                </tbody>
                                            </table>
                                        </div>

                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="quality-bg-2 text-center text-white h5">
                                                            Quality</th>
                                                        <th scope="col" class="owner-bg-2 text-center text-white h5">
                                                            Owner</th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">Final Report + Estimation/ Data to
                                                            customer</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table class="colortableparent" data-tag="1{{ $sa->id }}4" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="final_rep_estimation_data_to_customer_actual_date">

                                                                        <span class="<?php echo Colorizer($sa->final_rep_estimation_data_to_customer_actual_date, $sa->final_rep_estimation_data_to_customer_actual_qty, $sa->final_rep_estimation_data_to_customer_target_date, $sa->final_rep_estimation_data_to_customer_target_qty); ?>">{{ NullCheck($sa->final_rep_estimation_data_to_customer_actual_date) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="final_rep_estimation_data_to_customer_target_date">
                                                                        <span>{{ NullCheck($sa->final_rep_estimation_data_to_customer_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="final_rep_estimation_data_to_customer_actual_qty">

                                                                        <span class="<?php echo Colorizer($sa->final_rep_estimation_data_to_customer_actual_date, $sa->final_rep_estimation_data_to_customer_actual_qty, $sa->final_rep_estimation_data_to_customer_target_date, $sa->final_rep_estimation_data_to_customer_target_qty); ?>">{{ NullCheck($sa->final_rep_estimation_data_to_customer_actual_qty) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="final_rep_estimation_data_to_customer_target_qty">
                                                                        <span>{{ NullCheck($sa->final_rep_estimation_data_to_customer_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach

                                                </tbody>
                                            </table>
                                        </div>

                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="owner-bg text-center text-white h5">
                                                            Owner</th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">Client Update <br>date</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table style="width:100%">

                                                                <tr>
                                                                    <td class="ps-4 tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="client_update_date_05">
                                                                        {{ NullCheck($sa->client_update_date_05) }}
                                                                    </td>
                                                                </tr>

                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach
                                                </tbody>
                                            </table>
                                        </div>

                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="finance-bg text-center text-white h5">
                                                            Finance</th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">Cash Collected</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table style="width:100%">
                                                                <tr>
                                                                    <td class="ps-4 tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="cash_collected">
                                                                        {{ NullCheck($sa->cash_collected) }}
                                                                    </td>
                                                                </tr>

                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach
                                                </tbody>
                                            </table>
                                        </div>

                                        <div id="table" class="table-editable">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="production-bg text-center text-white h5">
                                                            Production
                                                        </th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">Final Delivery date</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table class="colortableparent" data-tag="1{{ $sa->id }}5" style="width:100%">
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="final_delivery_actual_date">

                                                                        <span class="<?php echo Colorizer($sa->final_delivery_actual_date, $sa->final_delivery_actual_qty, $sa->final_delivery_target_date, $sa->final_delivery_target_qty); ?>">{{ NullCheck($sa->final_delivery_actual_date) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-b border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="final_delivery_target_date">
                                                                        <span>{{ NullCheck($sa->final_delivery_target_date) }}</span>
                                                                    </td>
                                                                </tr>
                                                                <tr>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $canall }}" data-id="{{ $sa->id }}" data-col_name="final_delivery_actual_qty">

                                                                        <span class="<?php echo Colorizer($sa->final_delivery_actual_date, $sa->final_delivery_actual_qty, $sa->final_delivery_target_date, $sa->final_delivery_target_qty); ?>">{{ NullCheck($sa->final_delivery_actual_qty) }}</span>


                                                                    </td>
                                                                    <td class="pt-3-half tbl_cell four-col border-r" contenteditable="{{ $cantarget }}" data-id="{{ $sa->id }}" data-col_name="final_delivery_target_qty">
                                                                        <span>{{ NullCheck($sa->final_delivery_target_qty) }}</span>
                                                                    </td>
                                                                </tr>
                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach

                                                </tbody>
                                            </table>
                                        </div>

                                        <div id="table" class="table-editable action-table">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="gray-bg text-center text-white h5">
                                                            All departments</th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">Actions</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table style="width:100%">

                                                                <tr>
                                                                    <td class="ps-4 tbl_cell scroll-y" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="actions">
                                                                        @foreach (explode(',', $sa->actions) as $ac)
                                                                        <li>{{ $ac }}</li>
                                                                        @endforeach
                                                                        {{-- {{$sa->actions}} --}}
                                                                    </td>
                                                                </tr>

                                                                {{-- @foreach (explode(',', $row->data) as $fields)
                                                                <li>{{$fields}}</li>
                                                                @endforeach --}}

                                                            </table>
                                                        </td>
                                                    </tr>
                                                    @endforeach
                                                </tbody>
                                            </table>
                                        </div>

                                        <div id="table" class="table-editable action-table">
                                            <table class="table pb-0 mb-0">
                                                <thead class="black white-text">
                                                    <tr>
                                                        <th scope="col" class="graydark-bg text-center text-white h5">
                                                            All
                                                            departments</th>
                                                    </tr>
                                                </thead>
                                            </table>

                                            <table class="table table-bordered table-responsive-md table-striped text-center th-head">
                                                <thead>
                                                    <tr class="tbl-cell">
                                                        <th class="text-center">Lessons learnd </th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @foreach ($sales as $sa)
                                                    <tr class="tbl-cell">
                                                        <td class="p-0">
                                                            <table style="width:100%">

                                                                <tr class="">
                                                                    <td class="ps-4 tbl_cell" contenteditable="true" data-id="{{ $sa->id }}" data-col_name="lessons_learnd">
                                                                        {{ NullCheck($sa->lessons_learnd) }}
                                                                    </td>
                                                                    {{-- <td class="pt-3-half tbl_cell "
                                                                        contenteditable="true" data-id="{{ $sa->id }}"
                                                                    data-col_name="salesman">
                                                                    {{ NullCheck($sa->salesman) }}
                                                        </td> --}}
                                                    </tr>

                                            </table>
                                            </td>
                                            </tr>
                                            @endforeach
                                            </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                                <button class="float-end full-view" onclick="var el = document.getElementById('element'); el.requestFullscreen();">
                                    <i class="bi bi-fullscreen"></i> <br>Full view
                                </button>
                            </div>
                        </div>
                    </div>
                    <!-- Editable table -->
                    @else
                    <h1> No Sales</h1>
                    @endif
                </div>

            </div><!-- End Left side columns -->
        </div>

    </section>

</main><!-- End #main -->

<script src="https://code.jquery.com/jquery-3.7.0.min.js" integrity="sha256-2Pmvv0kuTBOenSvLm6bvfBSSHrUJ+3A7x6P5Ebd07/g=" crossorigin="anonymous"></script>
<script src="/assets/js/jquery.timeago.js" type="text/javascript"></script>
<script src="/assets/js/jquery-dateformat.min.js" type="text/javascript"></script>
<script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>



<script>
    function Colorizer(actualDate, tagetDate, actualQuantity, targetQuantity) {
        if (actualDate == "N/A" || tagetDate == "N/A" || actualQuantity == "N/A" || targetQuantity == "N/A") {
            return 'defaultblack';
        }
        if (Date.parse(actualDate) < Date.parse(tagetDate)) {
            if (actualQuantity == targetQuantity) {
                return 'equalgreen';
            } else {
                return 'beforeblue';
            }
        } else if (Date.parse(actualDate) == Date.parse(tagetDate)) {
            return 'equalgreen';
        } else if (Date.parse(actualDate) > Date.parse(tagetDate)) {
            return 'latered';
        }

    }

    function setColor(table) {

        var td = $('.colortableparent[data-tag="' + table + '"]').find('td')
        var color = Colorizer(td.eq(0).text().trim(), td.eq(1).text().trim(), td.eq(2).text().trim(), td.eq(3).text()
            .trim())
        td.eq(0).find('span').removeClass("defaultblack beforeblue equalgreen latered").addClass(color)
        td.eq(2).find('span').removeClass("defaultblack beforeblue equalgreen latered").addClass(color)

    }

        function updateTime() {
            $.ajax({
                url: "/last-update",
                type: "GET",
                headers: {
                    "X-CSRF-TOKEN": $('meta[name="csrf-token"]').attr(
                        "content"
                    ),
                    Accept: "application/json",
                },
                success: function(response) {
                    $('.last_update_duration').text("Last update:" + response.last_update_duration)
                    $('.las_uptdate_date').text(response.last_update_date)
                },
                error: function() {},
            });
        }
    updateTime()

    var debounce = function(func, wait, immediate) {
        var timeout;
        return function() {
            var context = this,
                args = arguments;
            var later = function() {
                timeout = null;
                if (!immediate) func.apply(context, args);
            };
            var callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func.apply(context, args);
        };
    };
    $(".btn_delete_sale").click(
        function() {
            Swal.fire({
                title: 'Are you sure?',
                text: "You won't be able to revert this!",
                icon: 'question',
                showCancelButton: true,
                confirmButtonColor: '#3085d6',
                cancelButtonColor: '#d33',
                confirmButtonText: 'Yes, delete it!'
            }).then((result) => {
                if (result.isConfirmed) {
                    //data to submit
                    var data = {
                        id: $(this).data("id"),
                    };

                    $.ajax({
                        url: "/delete-sale",
                        type: "DELETE",
                        headers: {
                            "X-CSRF-TOKEN": $('meta[name="csrf-token"]').attr(
                                "content"
                            ),
                            Accept: "application/json",
                        },
                        data: data,
                        success: function(response) {
                            location.reload();
                        },
                        error: function() {
                            Swal.fire(
                                'Error!',
                                'Delete Sale Error',
                                'error'
                            )
                        },
                    });
                }
            })
        }
    );
    $(".btn_undo_sale").click(
        function() {
            Swal.fire({
                title: 'Are you sure?',
                text: "You want to Undo ?",
                icon: 'question',
                showCancelButton: true,
                confirmButtonColor: '#3085d6',
                cancelButtonColor: '',
                confirmButtonText: 'Yes, undo it!'
            }).then((result) => {
                if (result.isConfirmed) {
                    //data to submit
                    var data = {
                        id: $(this).data("id"),
                    };

                    $.ajax({
                        url: "/undo-sale/" + data.id,
                        type: "GET",
                        headers: {
                            "X-CSRF-TOKEN": $('meta[name="csrf-token"]').attr(
                                "content"
                            ),
                            Accept: "application/json",
                        },
                        success: function(response) {
                            location.reload();
                        },
                        error: function(xhr, ajaxOptions, thrownError) {
                            if (xhr.status == 404) {
                                Swal.fire({
                                    toast: true,
                                    icon: 'error',
                                    title: 'Nothing More to Undo',
                                    animation: false,
                                    position: 'top',
                                    showConfirmButton: false,
                                    timer: 3000,
                                    timerProgressBar: true,
                                    didOpen: (toast) => {
                                        toast.addEventListener('mouseenter', Swal
                                            .stopTimer)
                                        toast.addEventListener('mouseleave', Swal
                                            .resumeTimer)
                                    }
                                })
                            } else {
                                Swal.fire(
                                    'Error!',
                                    'Undo Sale Error',
                                    'error'
                                )
                            }
                        },
                    });
                }
            })
        }
    );

    $('.tbl_cell').on('focus', function() {
        var range = document.createRange();
        range.selectNodeContents(this);
        var selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
    });

    $(".tbl_cell").keyup(debounce(function(e) {

        //data to submit
        var sales_data = {
            id: $(this).data("id"),
            col_name: $(this).data("col_name"),
            value: $.trim($(this).text()),
        };

        var tag = $(this).closest('.colortableparent').data("tag")
        if (!sales_data.value.includes("N/A")) {
            $.ajax({
                url: "/live_update_sale",
                type: "PUT",
                headers: {
                    "X-CSRF-TOKEN": $('meta[name="csrf-token"]').attr(
                        "content"
                    ),
                    Accept: "application/json",
                },
                data: sales_data,

                success: function(response) {
                    if (sales_data.value == '') {
                        $("td[data-col_name='" + sales_data.col_name + "']").html(
                            '<span class="defaultblack">N/A<span>');
                    }
                    setColor(tag)
                    updateTime()
                    Swal.fire({
                        toast: true,
                        icon: 'success',
                        title: 'Record Updated Successfully',
                        animation: false,
                        position: 'top',
                        showConfirmButton: false,
                        timer: 3000,
                        timerProgressBar: true,
                        didOpen: (toast) => {
                            toast.addEventListener('mouseenter', Swal.stopTimer)
                            toast.addEventListener('mouseleave', Swal.resumeTimer)
                        }
                    })
                },
                error: function() {
                    Swal.fire({
                        toast: true,
                        icon: 'error',
                        title: 'Update Sale Error',
                        animation: false,
                        position: 'top',
                        showConfirmButton: false,
                        timer: 3000,
                        timerProgressBar: true,
                        didOpen: (toast) => {
                            toast.addEventListener('mouseenter', Swal.stopTimer)
                            toast.addEventListener('mouseleave', Swal.resumeTimer)
                        }
                    })
                },
            });

        } else {}

    }, 500));


    $(".country_list").change(function(e) {

        //data to submit
        var sales_data = {
            id: $(this).data("id"),
            col_name: 'country_id',
            value: $.trim($(this).find(":selected").val()),
        };

        $.ajax({
            url: "/live_update_sale",
            type: "PUT",
            headers: {
                "X-CSRF-TOKEN": $('meta[name="csrf-token"]').attr(
                    "content"
                ),
                Accept: "application/json",
            },
            data: sales_data,
            success: function(response) {

            },
            error: function() {
                Swal.fire(
                    'Error!',
                    'Update Sale Error',
                    'error'
                )
            },
        });
    });
</script>


@endsection('content')