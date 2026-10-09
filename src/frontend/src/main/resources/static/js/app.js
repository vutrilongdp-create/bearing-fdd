$(document).ready(function() {

	if ($("#loggedUserFlag").val() != "admin") {
		$("#adminUsersBtt").hide();
	}

	var userLang;
	if ($("#webAppTitle").text() == "Detection and Diagnosis of Bearing Failures on Electric Motors") {
		userLang = 'en';
	} else {
		userLang = 'es-ES';
	}

	if ($("#loggedUserFlag").val() != "admin" && $("#loggedUserFlag").val() != "") {
		if (userLang == 'es-ES') {
			$("#manualLinkEN").hide();
		} else {
			$("#manualLinkEN").show();
		}

		if (userLang == 'en') {
			$("#manualLinkES").hide();
		} else {
			$("#manualLinkES").show();
		}
	}
	if ($("#loggedUserFlag").val() == "admin") {
		if (userLang == 'es-ES') {
			$("#manualAdminLinkEN").hide();
		} else {
			$("#manualAdminLinkEN").show();
		}

		if (userLang == 'en') {
			$("#manualAdminLinkES").hide();
		} else {
			$("#manualAdminLinkES").show();
		}
	}
	
	const dropContainer = $("#dropcontainer");
	const fileInput = $("#file");

	dropContainer.on("dragover", function(e) {
		e.preventDefault();
	});

	dropContainer.on("dragenter", function() {
		dropContainer.addClass("drag-active");
	});

	dropContainer.on("dragleave", function() {
		dropContainer.removeClass("drag-active");
	});

	dropContainer.on("drop", function(e) {
		e.preventDefault();
		dropContainer.removeClass("drag-active");
		fileInput.prop("files", e.originalEvent.dataTransfer.files);
	});

	$('.single-checkbox').click(function() {
		$('.single-checkbox').prop('checked', false);
		$(this).prop('checked', true);
	});

	$('#submitButton').click(function() {
		var selectedCheckbox = $('.single-checkbox:checked');
		var formToSubmit = selectedCheckbox.closest('form');

		if (selectedCheckbox.length === 1) {
			formToSubmit.submit();
		} else {
			$('#myModal').modal({
				backdrop: 'static',
				keyboard: false
			}).modal('show');
		}
	});


	$(".onlyNums").on("input", function(e) {
		var currentValue = $(this).val();

		var newValue = currentValue.replace(/[^0-9.,]/g, '');

		$(this).val(newValue);
	});
	
	$(".onlyNumsPlus").on("input", function(e) {
		var currentValue = $(this).val();

		var newValue = currentValue.replace(/\D/g, '');

		$(this).val(newValue);
	});

	if ($("#fault_detected").val() == 'true') {
		var faultTypes = $("#fault_type").val();

		var faultTypesArray = faultTypes.split(',');

		if ($("#fault_info").val() == 'A fault has been detected in an early stage') {
			$(".circle").each(function() {
				var circleId = $(this).attr("id");
				var circleType = getCircleType(circleId);

				if (faultTypesArray.includes(circleType)) {
					$(this).css("background-color", "yellow");
				}
			});
		}
		if ($("#fault_info").val() == 'A fault has been detected in a medium stage') {
			$(".circle").each(function() {
				var circleId = $(this).attr("id");
				var circleType = getCircleType(circleId);

				if (faultTypesArray.includes(circleType)) {
					$(this).css("background-color", "orange");
				}
			});
		}
		if ($("#fault_info").val() == 'A fault has been detected in a last degradation stage') {
			$(".circle").each(function() {
				var circleId = $(this).attr("id");
				var circleType = getCircleType(circleId);

				if (faultTypesArray.includes(circleType)) {
					$(this).css("background-color", "red");
				}
			});
		}

		$("#pdf1").css("display", "inline");
		$("#pdf2").css("display", "inline");
		$("#imgph1").hide();
		$("#imgph2").hide();
		$("#n_healthy_data").val($("#n_healthy_used").val());
		$("#n_healthy_data_upload").val($("#n_healthy_used").val());
	}

	if ($("#fault_detected").val() == 'false') {
		$("#imgph1").hide();
		$("#imgph2").hide();
		$("#n_healthy_data").val($("#n_healthy_used").val());
		$("#n_healthy_data_upload").val($("#n_healthy_used").val());
	}

	if ($("#files_added").val() == 1) {
		$("#formNewUpload").hide();
		$("#formDataCheckNew").show();
		$("#formDataCheckNewDos").show();
		$("#deleteDatasetFakeBtt").show();
		$("#semaforoDiv").show();
		$("#allImagesDiv").show();
		$("#xaiImagesDiv").show();
		$("#runNew").show();
	}
	if ($("#files_added").val() == 0) {
		$("#formNewUpload").show();
		$("#formDataCheckNew").hide();
		$("#formDataCheckNewDos").hide();
		$("#semaforoDiv").hide();
		$("#allImagesDiv").hide();
		$("#xaiImagesDiv").hide();
		$("#runNew").hide();
		$("#deleteDatasetFakeBtt").hide();
	}

	if ($("#warningsDivNew").is(":visible")) {
		$("#formNewUpload").hide();
		$("#formDataCheckNew").hide();
		$("#formDataCheckNewDos").hide();
		$("#deleteDatasetFakeBtt").hide();
	}

	if ($("#warningsDivAPI").is(":visible")) {
		$("#saveDatasetBtt").hide();
		$("#manageLink").hide();
		$("#adminUsersBtt").hide();
		$("#submitButton").hide();
		$("#newDatasetBtt").hide();
	}

	$("#id2send").val($("#id").val());
	$("#name2send").val($("#nombre").val());

	if ($("#imgOk").is(":visible")) {
		$("#imgph1").hide();
		$("#imgph2").hide();
	}

	actualizarEstadoBoton();
});

$(document).on('keydown', function(event) {
	if ($('#loadingModal').is(':visible') && event.keyCode === 27) {
		event.preventDefault();
		event.stopPropagation();
	}
});

function eliminarUser(item) {
	if (item != null && item != "admin") {
		jQuery.ajax({
			url: '/webAppMotorElectrico/deleteUser',
			type: 'POST',
			data: { item: item },
			success: function() {
				location.reload();
			}
		});
	}
}

function eliminarItem(item) {
	if (item != null) {
		jQuery.ajax({
			url: '/webAppMotorElectrico/deleteDataset',
			type: 'POST',
			data: { item: item },
			success: function() {
				location.reload();
			}
		});
	}
}

function eliminarSample(nombre, id) {
	if (nombre != null && id != null) {
		jQuery.ajax({
			url: '/webAppMotorElectrico/deleteSample',
			type: 'POST',
			data: { nombre: nombre, id: id },
			success: function() {
				location.reload();
			}
		});
	}
}

function actualizarEstadoBoton() {
	if ($('#regularNewAnalisys').is(':checked') || $('#upload4NewAnalisys').is(':checked')) {
		$('#runNew').prop('disabled', false);
	} else {
		$('#runNew').prop('disabled', true);
	}
}

$("#saveDatasetBtt").click(function() {
	$('#uploadingModal').modal({
		backdrop: 'static',
		keyboard: false
	}).modal('show');
	$("#formUpload").submit();
});

$("#uploadDataSamples").click(function() {
	$('#uploadingModal').modal({
		backdrop: 'static',
		keyboard: false
	}).modal('show');
	$("#formNewUpload").submit();
});

$("#saveDatasetInfoBtt").click(function() {
	if ($("#nombre").val() == "New Dataset") {
		$("#dataInfoAlreadyExistsH4").text($("#tmp4Exist").text());
		$("#warningsDivNew").css({ "background-color": "red", "border": "red" });
		return false;
	}
	if($("#shaft_frequency").val() == 0.0){
		$("#dataInfoAlreadyExistsH4").text($("#tmp4Zero").text());
		$("#warningsDivNew").css({ "background-color": "red", "border": "red" });
		return false;
	}
});

$("#runPre").click(function() {
	var nombre = $("#nombre").val();
	var shaftFrequency = $("#shaft_frequency").val();
	var samplingFrequency = $("#sampling_frequency").val();
	var bpfo = $("#bpfo").val();
	var bpfi = $("#bpfi").val();
	var bsf = $("#bsf").val();
	var ftf = $("#ftf").val();
	var firstSample = $("#min_to_check").val();
	var analyzedNumber = $("#max_to_check").val();
	var n_healthy = $("#n_healthy_data").val();

	if (n_healthy != '' && firstSample != '' && analyzedNumber != '') {
		$("#nombre_req").val(nombre);
		$("#shaft_frequency_req").val(shaftFrequency);
		$("#sampling_frequency_req").val(samplingFrequency);
		$("#bpfo_req").val(bpfo);
		$("#bpfi_req").val(bpfi);
		$("#bsf_req").val(bsf);
		$("#ftf_req").val(ftf);
		$("#first_sample_req").val(firstSample);
		$("#analyzed_number_req").val(analyzedNumber - firstSample);
		$("#healthy_number_req").val(n_healthy);

		$('#loadingModal').modal({
			backdrop: 'static',
			keyboard: false
		}).modal('show');

		$("#runPreform").submit();
	} else {
		location.reload();
	}
});

$('#regularNewAnalisys').change(function() {
	if ($(this).is(':checked')) {
		$('#upload4NewAnalisys').prop('checked', false);
	}
	actualizarEstadoBoton()
});

$('#upload4NewAnalisys').change(function() {
	if ($(this).is(':checked')) {
		$('#regularNewAnalisys').prop('checked', false);
	}
	actualizarEstadoBoton()
});


$("#runNew").click(function(event) {
	var nombre = $("#nombre").val();
	var shaftFrequency = $("#shaft_frequency").val();
	var samplingFrequency = $("#sampling_frequency").val();
	var bpfo = $("#bpfo").val();
	var bpfi = $("#bpfi").val();
	var bsf = $("#bsf").val();
	var ftf = $("#ftf").val();
	var firstSample = $("#min_to_check").val();
	var analyzedNumber = $("#max_to_check").val();
	var n_healthy = $("#n_healthy_data").val();

	if ($('#regularNewAnalisys').is(':checked')) {
		if (n_healthy != '' && firstSample != '' && analyzedNumber != '') {
			$("#nombre_req").val(nombre);
			$("#shaft_frequency_req").val(shaftFrequency);
			$("#sampling_frequency_req").val(samplingFrequency);
			$("#bpfo_req").val(bpfo);
			$("#bpfi_req").val(bpfi);
			$("#bsf_req").val(bsf);
			$("#ftf_req").val(ftf);
			$("#first_sample_req").val(firstSample);
			$("#analyzed_number_req").val(analyzedNumber - firstSample);
			$("#healthy_number_req").val(n_healthy);

			$('#loadingModal').modal({
				backdrop: 'static',
				keyboard: false
			}).modal('show');

			$("#runNewform").submit();
		} else {
			event.preventDefault();
		}
	}
	if ($('#upload4NewAnalisys').is(':checked')) {
		n_healthy = $("#n_healthy_data_upload").val();
		if (n_healthy != '') {
			$("#id_upp").val($("#id").val());
			$("#nombre_req_upp").val(nombre);
			$("#shaft_frequency_req_upp").val(shaftFrequency);
			$("#sampling_frequency_req_upp").val(samplingFrequency);
			$("#bpfo_req_upp").val(bpfo);
			$("#bpfi_req_upp").val(bpfi);
			$("#bsf_req_upp").val(bsf);
			$("#ftf_req_upp").val(ftf);
			$("#healthy_number_req_upp").val(n_healthy);

			$('#loadingModal').modal({
				backdrop: 'static',
				keyboard: false
			}).modal('show');

			$("#formDataCheckNewDos").submit();
		}
	}
});

function getCircleType(circleId) {
	switch (circleId) {
		case "colorCircle1":
			return "Outer_race";
		case "colorCircle2":
			return "Inner_race";
		case "colorCircle3":
			return "Bearing_Balls";
		case "colorCircle4":
			return "Cage";
		default:
			return "";
	}
}

$("#pdf1").click(function() {
	var fault_detected = $("#fault_detected").val();
	var fault_info = $("#fault_info").val();
	var fault_type = $("#fault_type").val();
	var fault_details = $("#fault_details").val();
	var analysis_result = $("#analysis_result").val();
	var resultTimeReport = parseStringToList($("#resultTimeReport").val());
	var resultFreqReport = parseStringToList($("#resultFreqReport").val());

	$.ajax({
		type: "GET",
		url: "/generateExp",
		data: {
			fault_detected: fault_detected,
			fault_info: fault_info,
			fault_type: fault_type,
			fault_details: fault_details,
			analysis_result: analysis_result,
			resultTimeReport: resultTimeReport,
			resultFreqReport: resultFreqReport
		},
		success: function(data) {
			var blob = b64toBlob(data, 'application/pdf');
			var link = document.createElement('a');
			link.href = window.URL.createObjectURL(blob);
			link.target = '_blank';
			document.body.appendChild(link);
			link.click();
			document.body.removeChild(link);
		},
		error: function(error) {
			console.error("Error al generar el PDF: " + error);
		}
	});
});

$("#pdf2").click(function() {
	var fault_detected = $("#fault_detected").val();
	var fault_info = $("#fault_info").val();
	var fault_type = $("#fault_type").val();
	var fault_details = $("#fault_details").val();
	var analysis_result = $("#analysis_result").val();
	var details_out = parseStringToList($("#fault_details_Outer_race").val());
	var details_in = parseStringToList($("#fault_details_Inner_race").val());
	var details_balls = parseStringToList($("#fault_details_Bearing_Balls").val());
	var details_cage = parseStringToList($("#fault_details_Cage").val());

	if (details_out == "") {
		details_out = [];
	}
	if (details_in == "") {
		details_in = [];
	}
	if (details_balls == "") {
		details_balls = [];
	}
	if (details_cage == "") {
		details_cage = [];
	}

	var data = {
		fault_detected: fault_detected,
		fault_info: fault_info,
		fault_type: fault_type,
		fault_details: fault_details,
		analysis_result: analysis_result,
		details_out: details_out,
		details_in: details_in,
		details_balls: details_balls,
		details_cage: details_cage
	};

	$.ajax({
		type: "POST",
		url: "/generateInt",
		contentType: "application/json",
		data: JSON.stringify(data),
		success: function(data) {
			var blob = b64toBlob(data, 'application/pdf');
			var link = document.createElement('a');
			link.href = window.URL.createObjectURL(blob);
			link.target = '_blank';
			document.body.appendChild(link);
			link.click();
			document.body.removeChild(link);
		},
		error: function(error) {
			console.error("Error al generar el PDF: " + error);
		}
	});
});

$("#min_to_check").click(function() {
	$(this).val('');
});

function b64toBlob(base64, contentType) {
	contentType = contentType || '';
	const sliceSize = 512;
	const byteCharacters = atob(base64);
	const byteArrays = [];

	for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
		const slice = byteCharacters.slice(offset, offset + sliceSize);
		const byteNumbers = new Array(slice.length);
		for (let i = 0; i < slice.length; i++) {
			byteNumbers[i] = slice.charCodeAt(i);
		}
		const byteArray = new Uint8Array(byteNumbers);
		byteArrays.push(byteArray);
	}

	return new Blob(byteArrays, { type: contentType });
}

function parseStringToList(string) {
	var trimmedString = string.trim().slice(1, -1);
	var elements = trimmedString.split(',');
	var list = elements.map(function(element) {
		return parseFloat(element.trim());
	});

	return list;
}
